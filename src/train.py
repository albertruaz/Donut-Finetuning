import argparse
import json
import math
import os
import sys
import glob
import datetime
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    set_seed,
)

from src.datasets import DonutDataCollator, DonutJsonlDataset
from utils.config_manager import RunPaths, load_config, make_run_dirs
from utils.logging_utils import setup_logger


def _safe_json_loads(text: str) -> Any:
    if not text:
        return {}
    try:
        # NaN 값을 null로 치환
        text = text.replace(": NaN", ": null").replace(":NaN", ":null")
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _strip_prompt(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


def build_compute_metrics(processor, prompt: str):
    pad_token_id = processor.tokenizer.pad_token_id
    
    # step 카운터를 위한 global 변수
    global validation_step_counter
    if 'validation_step_counter' not in globals():
        validation_step_counter = 0

    def compute_metrics(eval_preds):
        try:
            print(f"[VALIDATION DEBUG] Starting compute_metrics")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            print(f"[VALIDATION DEBUG] GPU Memory after cache clear: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            predictions, labels = eval_preds
            print(f"[VALIDATION DEBUG] Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
            print(f"[VALIDATION DEBUG] Labels shape: {labels.shape if hasattr(labels, 'shape') else type(labels)}")
            
            if isinstance(predictions, tuple):
                predictions = predictions[0]
                print(f"[VALIDATION DEBUG] Extracted predictions shape: {predictions.shape}")

            print(f"[VALIDATION DEBUG] Starting batch decode...")
            
            # 음수 토큰 ID 처리 - predictions에서 음수값들을 패드 토큰으로 변경
            predictions_clean = np.where(predictions < 0, pad_token_id, predictions)
            print(f"[VALIDATION DEBUG] Cleaned predictions shape: {predictions_clean.shape}")
            print(f"[VALIDATION DEBUG] Predictions min/max: {predictions_clean.min()}/{predictions_clean.max()}")
            
            decoded_preds = processor.batch_decode(predictions_clean, skip_special_tokens=True)
            print(f"[VALIDATION DEBUG] Decoded {len(decoded_preds)} predictions")
            
            labels = np.where(labels != -100, labels, pad_token_id)
            print(f"[VALIDATION DEBUG] Labels shape after cleaning: {labels.shape}")
            print(f"[VALIDATION DEBUG] Labels min/max: {labels.min()}/{labels.max()}")
            
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
            print(f"[VALIDATION DEBUG] Decoded {len(decoded_labels)} labels")

            decoded_preds = [_strip_prompt(pred.strip(), prompt) for pred in decoded_preds]
            decoded_labels = [_strip_prompt(label.strip(), prompt) for label in decoded_labels]

            print(f"[VALIDATION DEBUG] Starting JSON parsing...")
            pred_objs = [_safe_json_loads(pred) for pred in decoded_preds]
            label_objs = [_safe_json_loads(label) for label in decoded_labels]
            print(f"[VALIDATION DEBUG] Parsed {len(pred_objs)} predictions and {len(label_objs)} labels")

            total = len(pred_objs)
            exact = 0
            field_totals: Dict[str, int] = {}
            field_correct: Dict[str, int] = {}

            for pred, gold in zip(pred_objs, label_objs):
                try:
                    if isinstance(pred, dict) and isinstance(gold, dict):
                        if pred == gold:
                            exact += 1
                        keys = set(pred.keys()) | set(gold.keys())
                        for key in keys:
                            field_totals[key] = field_totals.get(key, 0) + 1
                            pred_val = pred.get(key, "")
                            gold_val = gold.get(key, "")
                            # None 값들을 빈 문자열로 처리
                            if pred_val is None:
                                pred_val = ""
                            if gold_val is None:
                                gold_val = ""
                            if pred_val == gold_val:
                                field_correct[key] = field_correct.get(key, 0) + 1
                    else:
                        if pred == gold:
                            exact += 1
                except Exception as e:
                    # 개별 샘플 처리 오류는 건너뛰기
                    print(f"Warning: Error processing sample: {e}")
                    continue

            metrics: Dict[str, float] = {}
            metrics["exact_match"] = exact / total if total else 0.0
            for key in field_totals:
                if field_totals[key] > 0:
                    metrics[f"acc_{key}"] = field_correct.get(key, 0) / field_totals[key]
            
            print(f"[VALIDATION DEBUG] Computed metrics: {metrics}")
            print(f"[VALIDATION DEBUG] GPU Memory after metrics: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            # 중간 validation에서도 샘플 저장 (첫 10개만)
            try:
                global validation_step_counter
                validation_step_counter += 1
                
                if hasattr(eval_preds, 'step') or True:  # 항상 실행
                    # 현재 실행 중인 run 폴더 찾기
                    import glob
                    run_dirs = glob.glob("results/run_*")
                    if run_dirs:
                        latest_run_dir = max(run_dirs, key=os.path.getctime)
                        
                        # 간단한 샘플 저장 (첫 10개만, 빠른 처리를 위해)
                        sample_predictions = []
                        num_samples = min(10, len(predictions))
                        
                        for i in range(num_samples):
                            pred_text = decoded_preds[i] if i < len(decoded_preds) else ""
                            label_text = decoded_labels[i] if i < len(decoded_labels) else ""
                            
                            sample_predictions.append({
                                "sample_id": i,
                                "prediction": pred_text,
                                "ground_truth": label_text,
                                "prediction_json": _safe_json_loads(pred_text),
                                "ground_truth_json": _safe_json_loads(label_text)
                            })
                        
                        # step과 타임스탬프가 포함된 파일명으로 저장
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%H%M%S")
                        step_estimate = validation_step_counter * 200  # eval_steps=200 기준
                        sample_file = os.path.join(latest_run_dir, f"validation_step{step_estimate:04d}_{timestamp}.json")
                        
                        with open(sample_file, "w", encoding="utf-8") as f:
                            json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
                        
                        print(f"[VALIDATION DEBUG] Saved {len(sample_predictions)} samples to {sample_file}")
            except Exception as e:
                print(f"[VALIDATION DEBUG] Could not save intermediate samples: {e}")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            print(f"[VALIDATION DEBUG] GPU Memory after final cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            return metrics
            
        except Exception as e:
            print(f"[VALIDATION ERROR] Error in compute_metrics: {e}")
            print(f"[VALIDATION ERROR] GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            import traceback
            print(f"[VALIDATION ERROR] Traceback: {traceback.format_exc()}")
            
            # 에러 시에도 메모리 정리
            torch.cuda.empty_cache()
            print(f"[VALIDATION ERROR] GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            
            # 오류 발생 시 기본 메트릭 반환
            return {"exact_match": 0.0, "error": 1.0}

    return compute_metrics


def maybe_add_special_tokens(processor, model, cfg: Dict[str, Any]) -> None:
    special_tokens = cfg.get("tokenizer", {}).get("additional_special_tokens", [])
    if special_tokens:
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if added > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))


def initialise_wandb(cfg: Dict[str, Any], paths: RunPaths, logger):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging is enabled in the config but the `wandb` package "
            "is not installed. Install it with `pip install wandb`."
        ) from exc

    env_map = {
        "WANDB_PROJECT": wandb_cfg.get("project"),
        "WANDB_ENTITY": wandb_cfg.get("entity"),
        "WANDB_RUN_GROUP": wandb_cfg.get("group"),
        "WANDB_NOTES": wandb_cfg.get("notes"),
        "WANDB_MODE": wandb_cfg.get("mode"),
        "WANDB_TAGS": ",".join(wandb_cfg.get("tags", [])) if wandb_cfg.get("tags") else None,
        "WANDB_DIR": paths.run_dir,
    }
    for key, value in env_map.items():
        if value:
            os.environ.setdefault(key, str(value))

    init_kwargs = dict(wandb_cfg.get("init_kwargs", {}))
    project = wandb_cfg.get("project")
    if project:
        init_kwargs.setdefault("project", project)
    entity = wandb_cfg.get("entity")
    if entity:
        init_kwargs.setdefault("entity", entity)
    name = wandb_cfg.get("name") or wandb_cfg.get("run_name")
    if name:
        init_kwargs.setdefault("name", name)
    group = wandb_cfg.get("group")
    if group:
        init_kwargs.setdefault("group", group)
    tags = wandb_cfg.get("tags")
    if tags:
        init_kwargs.setdefault("tags", tags)
    mode = wandb_cfg.get("mode")
    if mode:
        init_kwargs.setdefault("mode", mode)
    init_kwargs.setdefault("dir", paths.run_dir)
    init_kwargs.setdefault("config", cfg)

    allow_offline = bool(wandb_cfg.get("allow_offline_fallback", True))
    suppress_errors = bool(wandb_cfg.get("suppress_errors", True))

    def _init_with_kwargs(kwargs):
        return wandb.init(**kwargs)

    try:
        run = _init_with_kwargs(init_kwargs)
        logger.info(
            "Weights & Biases logging enabled: %s",
            getattr(run, "url", getattr(run, "name", "<unnamed>")),
        )
        return run
    except Exception as exc:  # noqa: BLE001 - surface W&B init errors gracefully
        logger.warning("W&B initialisation failed: %s", exc)
        if allow_offline and init_kwargs.get("mode", "online") != "offline":
            logger.warning("Retrying W&B initialisation in offline mode")
            offline_kwargs = dict(init_kwargs)
            offline_kwargs["mode"] = "offline"
            os.environ["WANDB_MODE"] = "offline"
            try:
                run = _init_with_kwargs(offline_kwargs)
                logger.info("W&B offline fallback enabled")
                return run
            except Exception as offline_exc:  # noqa: BLE001
                logger.error("W&B offline fallback failed: %s", offline_exc)
        if suppress_errors:
            logger.warning("Disabling W&B logging due to initialisation failure")
            return None
        raise


def _filter_training_kwargs(
    kwargs: Dict[str, Any], logger
) -> Tuple[Dict[str, Any], Set[str]]:
    try:
        valid_params = set(signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
    except (ValueError, TypeError):
        return kwargs, set()

    filtered = {}
    dropped: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in valid_params:
            filtered[key] = value
        else:
            dropped[key] = value

    if dropped:
        logger.warning(
            "Dropping unsupported training arguments: %s",
            ", ".join(sorted(dropped.keys())),
        )
    return filtered, set(dropped.keys())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json", help="Path to config JSON")
    return ap.parse_args()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = make_run_dirs(cfg["output_root"])
    logger = setup_logger(paths.logs_dir)

    wandb_run = initialise_wandb(cfg, paths, logger)
    wandb_enabled = wandb_run is not None

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info("Seed fixed to %d", seed)

    model_name = cfg["model_name"]
    processor = DonutProcessor.from_pretrained(model_name, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    maybe_add_special_tokens(processor, model, cfg)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
        )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    task_prompt = cfg.get("tokenizer", {}).get("task_prompt", "")
    data_cfg = cfg.get("data", {})
    max_target_length = int(data_cfg.get("max_target_length", 512))
    download_timeout = int(data_cfg.get("download_timeout", 10))

    train_dataset = DonutJsonlDataset(
        data_cfg["train_jsonl"],
        processor,
        task_prompt=task_prompt,
        max_target_length=max_target_length,
        image_root=data_cfg.get("image_root"),
        image_cache_dir=data_cfg.get("image_cache_dir"),
        download_timeout=download_timeout,
    )
    logger.info("Train samples: %d", len(train_dataset))
    if wandb_run is not None:
        wandb_run.summary["num_train_samples"] = len(train_dataset)

    valid_path = data_cfg.get("valid_jsonl")
    valid_dataset: Optional[DonutJsonlDataset] = None
    if valid_path:
        valid_dataset = DonutJsonlDataset(
            valid_path,
            processor,
            task_prompt=task_prompt,
            max_target_length=max_target_length,
            image_root=data_cfg.get("image_root"),
            image_cache_dir=data_cfg.get("image_cache_dir"),
            download_timeout=download_timeout,
        )
        logger.info("Validation samples: %d", len(valid_dataset))
        if wandb_run is not None:
            wandb_run.summary["num_validation_samples"] = len(valid_dataset)

    collator = DonutDataCollator(processor)

    train_cfg = cfg.get("train", {})
    generation_cfg = cfg.get("generation", {})
    metric_for_best_model = train_cfg.get("metric_for_best_model", "exact_match")

    report_to_cfg = train_cfg.get("report_to")
    if report_to_cfg is None:
        report_to = ["wandb"] if wandb_enabled else []
    elif isinstance(report_to_cfg, str):
        report_to = [report_to_cfg] if report_to_cfg != "none" else []
    else:
        report_to = list(report_to_cfg)
        if wandb_enabled and "wandb" not in report_to:
            report_to.append("wandb")
    
    # WandB 권한 오류 방지를 위해 강제로 비활성화
    if "none" in str(report_to_cfg).lower():
        report_to = []

    run_name = train_cfg.get("run_name")
    if not run_name:
        wandb_cfg = cfg.get("wandb", {})
        run_name = wandb_cfg.get("name") or wandb_cfg.get("run_name")

    training_kwargs: Dict[str, Any] = {
        "output_dir": paths.ckpt_dir,
        "num_train_epochs": train_cfg.get("num_train_epochs", 5),
        "per_device_train_batch_size": train_cfg.get(
            "per_device_train_batch_size", 2
        ),
        "per_device_eval_batch_size": train_cfg.get(
            "per_device_eval_batch_size", 2
        ),
        "learning_rate": train_cfg.get("learning_rate", 5e-5),
        "warmup_steps": train_cfg.get("warmup_steps", 0),
        "weight_decay": train_cfg.get("weight_decay", 0.0),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "eval_strategy": train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "steps"))
        if valid_dataset is not None
        else "no",
        "eval_steps": train_cfg.get("eval_steps", 200),
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_steps": train_cfg.get("save_steps", 200),
        "gradient_accumulation_steps": train_cfg.get(
            "gradient_accumulation_steps", 1
        ),
        "load_best_model_at_end": train_cfg.get("load_best_model_at_end", True),
        "fp16": train_cfg.get("fp16", False),
        "report_to": report_to,
        "save_total_limit": train_cfg.get("save_total_limit", 2),
        "logging_dir": paths.logs_dir,
        "seed": seed,
        "predict_with_generate": True,
        "generation_max_length": generation_cfg.get("max_length", max_target_length),
        "generation_num_beams": generation_cfg.get("num_beams", 1),
        "generation_do_sample": False,
        "remove_unused_columns": False,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": train_cfg.get("greater_is_better", True),
        "run_name": run_name,
    }

    training_kwargs, dropped_keys = _filter_training_kwargs(training_kwargs, logger)

    if "eval_strategy" not in training_kwargs:
        if train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy")) not in (None, "no") and "eval_strategy" in dropped_keys:
            logger.warning(
                "Current transformers version does not support evaluation strategy configuration; proceeding without periodic evaluation."
            )
        if training_kwargs.get("load_best_model_at_end"):
            logger.warning(
                "Disabling load_best_model_at_end because evaluation strategy is unavailable."
            )
            training_kwargs["load_best_model_at_end"] = False
            training_kwargs.pop("metric_for_best_model", None)
            training_kwargs.pop("greater_is_better", None)
        if training_kwargs.get("save_strategy") not in (None, "no"):
            logger.warning(
                "Setting save_strategy to 'no' because evaluation strategy is unavailable."
            )
            training_kwargs["save_strategy"] = "no"
        training_kwargs.pop("eval_steps", None)

    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    
    print(f"[TRAINING DEBUG] Training arguments created")
    print(f"[TRAINING DEBUG] eval_strategy: {training_args.eval_strategy}")
    print(f"[TRAINING DEBUG] eval_steps: {training_args.eval_steps}")
    print(f"[TRAINING DEBUG] do_eval: {training_args.do_eval}")
    print(f"[TRAINING DEBUG] valid_dataset: {valid_dataset is not None}")
    if valid_dataset:
        print(f"[TRAINING DEBUG] validation samples: {len(valid_dataset)}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor, task_prompt)
        if valid_dataset is not None
        else None,
    )
    
    print(f"[TRAINING DEBUG] Trainer created successfully")
    print(f"[TRAINING DEBUG] GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    logger.info("Starting training for %s", model_name)
    try:
        print(f"[TRAINING DEBUG] About to start trainer.train()")
        print(f"[TRAINING DEBUG] GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        
        # Custom training loop with step-by-step logging
        total_steps = len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        print(f"[TRAINING DEBUG] Expected total steps: {total_steps}")
        print(f"[TRAINING DEBUG] Evaluation will happen at step: {training_args.eval_steps}")
        
        trainer.train()
        print(f"[TRAINING DEBUG] Training completed successfully!")
        
    except Exception as e:
        print(f"[TRAINING ERROR] Training failed with error: {e}")
        print(f"[TRAINING ERROR] GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        import traceback
        print(f"[TRAINING ERROR] Traceback: {traceback.format_exc()}")
        raise
    finally:
        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

    logger.info("Saving model and processor to %s", paths.model_dir)
    model.save_pretrained(paths.model_dir)
    processor.save_pretrained(paths.model_dir)

    # Validation 샘플들에 대한 추론 결과 저장
    if valid_dataset is not None and len(valid_dataset) > 0:
        logger.info("Generating sample predictions for validation dataset...")
        try:
            sample_predictions = []
            num_samples = min(50, len(valid_dataset))
            
            model.eval()
            with torch.no_grad():
                for i in range(num_samples):
                    sample = valid_dataset[i]
                    original_record = valid_dataset.records[i]  # 원본 데이터 접근
                    
                    # Input 준비
                    pixel_values = sample['pixel_values'].unsqueeze(0).to(model.device)
                    
                    # Ground truth - 원본 JSON 객체에서 가져와서 clean하게 사용
                    original_output = original_record.get("output", {})
                    # NaN 값들을 빈 문자열로 정리
                    clean_ground_truth = {}
                    for key, value in original_output.items():
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            clean_ground_truth[str(key)] = ""
                        else:
                            clean_ground_truth[str(key)] = str(value)
                    
                    # 디코딩용 ground truth (기존 방식)
                    labels = sample['labels']
                    ground_truth_decoded = processor.batch_decode([labels], skip_special_tokens=True)[0]
                    ground_truth_decoded = _strip_prompt(ground_truth_decoded.strip(), task_prompt)
                    
                    # Image URL 추출
                    image_url = original_record.get("input", {}).get("image_url", "N/A")
                    
                    # 추론 실행
                    outputs = model.generate(
                        pixel_values,
                        max_length=max_target_length,
                        num_beams=generation_cfg.get("num_beams", 1),
                        early_stopping=generation_cfg.get("early_stopping", True)
                    )
                    
                    # 예측 결과 디코딩
                    prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    prediction = _strip_prompt(prediction.strip(), task_prompt)
                    
                    sample_predictions.append({
                        "sample_id": i,
                        "image_url": image_url,
                        "ground_truth": clean_ground_truth,  # Clean JSON 객체 사용
                        "prediction": prediction,
                        "ground_truth_json": clean_ground_truth,  # 이미 JSON 객체이므로 그대로 사용
                        "prediction_json": _safe_json_loads(prediction)
                    })
                    
                    # 진행률 표시 (10개마다)
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{num_samples} validation samples")
            
            # 결과를 JSON 파일로 저장
            predictions_file = os.path.join(paths.run_dir, "validation_samples.json")
            with open(predictions_file, "w", encoding="utf-8") as f:
                json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(sample_predictions)} validation sample predictions to {predictions_file}")
            
        except Exception as e:
            logger.error(f"Error generating validation samples: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # GPU 메모리 정리
            torch.cuda.empty_cache()

    metadata = {
        "config": cfg,
        "best_metric": getattr(trainer.state, "best_metric", None),
        "global_step": getattr(trainer.state, "global_step", None),
        "model_dir": paths.model_dir,
    }
    with open(os.path.join(paths.run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Run complete: %s", paths.run_dir)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    main(args.config)
