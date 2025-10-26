import argparse
import json
import os
import sys
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import (
    DonutProcessor,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    VisionEncoderDecoderModel,
    set_seed,
)

from src.datasets import DonutDataCollator, DonutJsonlDataset
from utils.config_manager import RunPaths, load_config, make_run_dirs
from utils.logging_utils import initialise_wandb, setup_logger


class EvalSampleLogger:
    """Persists a small slice of eval predictions for inspection."""

    def __init__(self, run_dir: str, max_samples: int = 10, step_interval: Optional[int] = None, prefix: str = "validation") -> None:
        self.run_dir = Path(run_dir)
        self.max_samples = max_samples
        self.counter = 0
        self.step_interval = step_interval if step_interval and step_interval > 0 else None
        self.prefix = prefix

    def save(self, predictions: List[str], labels: List[str]) -> None:
        if not predictions:
            return

        limit = min(self.max_samples, len(predictions), len(labels))
        if limit == 0:
            return

        samples = []
        for idx in range(limit):
            pred_text = predictions[idx]
            label_text = labels[idx]
            samples.append(
                {
                    "sample_id": idx,
                    "prediction": pred_text,
                    "ground_truth": label_text,
                    "prediction_json": _safe_json_loads(pred_text),
                    "ground_truth_json": _safe_json_loads(label_text),
                }
            )

        timestamp = datetime.now().strftime("%H%M%S")
        step_value = self.counter
        if self.step_interval is not None:
            step_value *= self.step_interval

        filename = self.run_dir / f"{self.prefix}_step{step_value:04d}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
        self.counter += 1


class TestEvaluationCallback(TrainerCallback):
    """Periodically evaluates test dataset during training."""

    def __init__(
        self,
        test_dataset,
        processor,
        task_prompt: str,
        test_sample_logger: Optional[EvalSampleLogger],
    ):
        self.test_dataset = test_dataset
        self.processor = processor
        self.task_prompt = task_prompt
        self.test_sample_logger = test_sample_logger
        self.last_eval_step = -1
        self.trainer = None  # Will be set after trainer creation

    def set_trainer(self, trainer):
        """Set the trainer reference."""
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        """Run test evaluation after each validation evaluation."""
        if self.test_dataset is None or self.trainer is None:
            return
        
        # Avoid duplicate evaluation for same step
        if state.global_step == self.last_eval_step:
            return
        
        self.last_eval_step = state.global_step
        
        print(f"[TEST EVAL] Running test evaluation at step {state.global_step}")

        # Temporarily swap datasets and compute_metrics
        original_eval_dataset = self.trainer.eval_dataset
        original_compute_metrics = self.trainer.compute_metrics

        self.trainer.eval_dataset = self.test_dataset
        self.trainer.compute_metrics = build_compute_metrics(
            self.processor, self.task_prompt, self.test_sample_logger
        )

        try:
            self.trainer.evaluate(metric_key_prefix="test")
            print(f"[TEST EVAL] Completed test evaluation at step {state.global_step}")
        except Exception as e:
            print(f"[TEST EVAL] Error during test evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original
            self.trainer.eval_dataset = original_eval_dataset
            self.trainer.compute_metrics = original_compute_metrics


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


def build_compute_metrics(processor, prompt: str, sample_logger: Optional[EvalSampleLogger] = None):
    pad_token_id = processor.tokenizer.pad_token_id
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # DEBUG: 첫 번째 샘플의 원본 토큰 확인
        print(f"\n[DEBUG] First prediction tokens (before cleaning): {predictions[0][:20]}")
        print(f"[DEBUG] Prediction shape: {predictions.shape}")
        
        predictions = np.where(predictions < 0, pad_token_id, predictions)
        labels = np.where(labels == -100, pad_token_id, labels)
        
        # DEBUG: 첫 번째 샘플의 정제된 토큰 확인
        print(f"[DEBUG] First prediction tokens (after cleaning): {predictions[0][:20]}")

        decoded_preds = [
            _strip_prompt(text.strip(), prompt)
            for text in processor.batch_decode(predictions, skip_special_tokens=True)
        ]
        decoded_labels = [
            _strip_prompt(text.strip(), prompt)
            for text in processor.batch_decode(labels, skip_special_tokens=True)
        ]
        
        # DEBUG: 첫 3개 샘플의 디코딩 결과
        print(f"[DEBUG] First 3 decoded predictions:")
        for i in range(min(3, len(decoded_preds))):
            print(f"  [{i}] pred: {repr(decoded_preds[i][:100])}")
            print(f"  [{i}] gold: {repr(decoded_labels[i][:100])}")

        pred_objs = [_safe_json_loads(text) for text in decoded_preds]
        label_objs = [_safe_json_loads(text) for text in decoded_labels]

        total = len(pred_objs)
        exact = 0
        field_totals: Dict[str, int] = {}
        field_correct: Dict[str, int] = {}

        for pred, gold in zip(pred_objs, label_objs):
            if isinstance(pred, dict) and isinstance(gold, dict):
                if pred == gold:
                    exact += 1
                keys = set(pred.keys()) | set(gold.keys())
                for key in keys:
                    field_totals[key] = field_totals.get(key, 0) + 1
                    pred_val = "" if pred.get(key) is None else pred.get(key)
                    gold_val = "" if gold.get(key) is None else gold.get(key)
                    if pred_val == gold_val:
                        field_correct[key] = field_correct.get(key, 0) + 1
            else:
                if pred == gold:
                    exact += 1

        metrics: Dict[str, float] = {
            "exact_match": exact / total if total else 0.0,
        }
        for key in sorted(field_totals):
            divisor = field_totals[key]
            if divisor:
                metrics[f"acc_{key}"] = field_correct.get(key, 0) / divisor

        if sample_logger is not None:
            sample_logger.save(decoded_preds, decoded_labels)
        return metrics

    return compute_metrics


def maybe_add_special_tokens(processor, model, cfg: Dict[str, Any]) -> None:
    special_tokens = cfg.get("tokenizer", {}).get("additional_special_tokens", [])
    if special_tokens:
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if added > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))


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

    test_path = data_cfg.get("test_jsonl")
    test_dataset: Optional[DonutJsonlDataset] = None
    if test_path:
        test_dataset = DonutJsonlDataset(
            test_path,
            processor,
            task_prompt=task_prompt,
            max_target_length=max_target_length,
            image_root=data_cfg.get("image_root"),
            image_cache_dir=data_cfg.get("image_cache_dir"),
            download_timeout=download_timeout,
        )
        logger.info("Test samples: %d", len(test_dataset))
        if wandb_run is not None:
            wandb_run.summary["num_test_samples"] = len(test_dataset)

    collator = DonutDataCollator(processor)

    train_cfg = cfg.get("train", {})
    generation_cfg = cfg.get("generation", {})
    metric_for_best_model = train_cfg.get("metric_for_best_model", "exact_match")

    sample_logger: Optional[EvalSampleLogger] = None
    test_sample_logger: Optional[EvalSampleLogger] = None
    eval_interval_raw = train_cfg.get("eval_steps", 200)
    try:
        eval_interval = int(eval_interval_raw)
    except (TypeError, ValueError):
        eval_interval = None
    
    if valid_dataset is not None:
        sample_logger = EvalSampleLogger(paths.run_dir, step_interval=eval_interval, prefix="validation")
    
    if test_dataset is not None:
        test_sample_logger = EvalSampleLogger(paths.run_dir, step_interval=eval_interval, prefix="test")

    prompt_token_ids: Optional[List[int]] = None
    if task_prompt:
        prompt_token_ids = processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
        ).input_ids
        if prompt_token_ids and isinstance(prompt_token_ids[0], list):
            prompt_token_ids = prompt_token_ids[0]

    generation_max_length = int(
        generation_cfg.get("max_length", max(max_target_length, 160))
    )
    generation_num_beams = int(generation_cfg.get("num_beams", 5))
    repetition_penalty = float(generation_cfg.get("repetition_penalty", 1.0))
    min_new_tokens_cfg = generation_cfg.get("min_new_tokens")
    min_new_tokens: Optional[int] = None
    if min_new_tokens_cfg is not None:
        try:
            min_new_tokens = int(min_new_tokens_cfg)
        except (TypeError, ValueError):
            min_new_tokens = None

    generation_config_kwargs = {
        "max_length": generation_max_length,
        "num_beams": generation_num_beams,
        "repetition_penalty": repetition_penalty,
        "early_stopping": generation_cfg.get("early_stopping", True),
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "decoder_start_token_id": model.config.decoder_start_token_id,
    }
    if min_new_tokens is not None:
        generation_config_kwargs["min_new_tokens"] = min_new_tokens

    generation_config = GenerationConfig(**generation_config_kwargs)

    if prompt_token_ids:
        # ✅ BOS는 0번 위치 유지, 프롬프트는 1번부터 강제
        bos_id = model.config.decoder_start_token_id
        if bos_id is None:
            bos_id = processor.tokenizer.bos_token_id or processor.tokenizer.cls_token_id
        
        forced_decoder_ids = [(0, int(bos_id))]
        forced_decoder_ids += [(i + 1, int(token)) for i, token in enumerate(prompt_token_ids)]
        generation_config.forced_decoder_ids = forced_decoder_ids
        
        # ❌ decoder_start_token_id는 절대 건드리지 않음 (BOS 유지)
        # generation_config.decoder_start_token_id = int(prompt_token_ids[0])  # 삭제
        # model.config.decoder_start_token_id = int(prompt_token_ids[0])       # 삭제

    model.generation_config = generation_config
    model.config.max_length = generation_config.max_length
    model.config.num_beams = generation_config.num_beams
    model.config.repetition_penalty = generation_config.repetition_penalty
    if generation_config.min_new_tokens is not None:
        model.config.min_new_tokens = generation_config.min_new_tokens

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
        "generation_max_length": generation_config.max_length,
        "generation_num_beams": generation_config.num_beams,
        "generation_do_sample": False,
        "remove_unused_columns": False,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": train_cfg.get("greater_is_better", True),
        "run_name": run_name,
    }

    if generation_config.min_new_tokens is not None:
        training_kwargs["generation_min_new_tokens"] = generation_config.min_new_tokens

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

    # Prepare callbacks
    callbacks = []
    test_callback = None
    if test_dataset is not None and test_sample_logger is not None:
        test_callback = TestEvaluationCallback(
            test_dataset=test_dataset,
            processor=processor,
            task_prompt=task_prompt,
            test_sample_logger=test_sample_logger,
        )
        callbacks.append(test_callback)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    compute_metrics=build_compute_metrics(processor, task_prompt, sample_logger)
    if valid_dataset is not None
    else None,
        tokenizer=processor.tokenizer,
        callbacks=callbacks if callbacks else None,
    )
    
    # Set trainer reference in callback
    if test_callback is not None:
        test_callback.set_trainer(trainer)
    
    print(f"[TRAINING DEBUG] Trainer created successfully")
    print(f"[TRAINING DEBUG] GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

    if valid_dataset is not None and training_args.do_eval:
        logger.info("Running initial evaluation at step 0 before training")
        trainer.evaluate(metric_key_prefix="initial")
        # ✅ test 평가는 TestEvaluationCallback이 자동으로 수행

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

    # Validation 샘플 예측 저장: trainer.predict를 그대로 활용
    if valid_dataset is not None and len(valid_dataset) > 0:
        logger.info("Generating sample predictions for validation dataset...")
        try:
            prediction_output = trainer.predict(valid_dataset, metric_key_prefix="val_samples")
            decoded_preds = [
                _strip_prompt(text.strip(), task_prompt)
                for text in processor.batch_decode(
                    prediction_output.predictions, skip_special_tokens=True
                )
            ]
            decoded_labels = [
                _strip_prompt(text.strip(), task_prompt)
                for text in processor.batch_decode(
                    prediction_output.label_ids, skip_special_tokens=True
                )
            ]

            sample_predictions = []
            num_samples = min(50, len(decoded_preds))
            for idx in range(num_samples):
                record = valid_dataset.records[idx]
                prediction_text = decoded_preds[idx]
                label_text = decoded_labels[idx]
                prediction_json = _safe_json_loads(prediction_text)
                label_json = _safe_json_loads(label_text)

                sample_predictions.append(
                    {
                        "sample_id": idx,
                        "image_path": record.get("input", {}).get("image_path"),
                        "image_url": record.get("input", {}).get("image_url"),
                        "prediction": prediction_text,
                        "ground_truth": label_text,
                        "prediction_json": prediction_json,
                        "ground_truth_json": label_json,
                    }
                )

            predictions_file = os.path.join(paths.run_dir, "validation_samples.json")
            with open(predictions_file, "w", encoding="utf-8") as f:
                json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
            logger.info(
                "Saved %d validation sample predictions to %s",
                len(sample_predictions),
                predictions_file,
            )
        except Exception as exc:  # noqa: BLE001 - surface prediction issues
            logger.error("Error generating validation samples: %s", exc)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

    # Test 샘플 예측 저장: trainer.predict를 그대로 활용
    if test_dataset is not None and len(test_dataset) > 0:
        logger.info("Generating sample predictions for test dataset...")
        try:
            prediction_output = trainer.predict(test_dataset, metric_key_prefix="test_samples")
            decoded_preds = [
                _strip_prompt(text.strip(), task_prompt)
                for text in processor.batch_decode(
                    prediction_output.predictions, skip_special_tokens=True
                )
            ]
            decoded_labels = [
                _strip_prompt(text.strip(), task_prompt)
                for text in processor.batch_decode(
                    prediction_output.label_ids, skip_special_tokens=True
                )
            ]

            sample_predictions = []
            num_samples = min(50, len(decoded_preds))
            for idx in range(num_samples):
                record = test_dataset.records[idx]
                prediction_text = decoded_preds[idx]
                label_text = decoded_labels[idx]
                prediction_json = _safe_json_loads(prediction_text)
                label_json = _safe_json_loads(label_text)

                sample_predictions.append(
                    {
                        "sample_id": idx,
                        "image_path": record.get("input", {}).get("image_path"),
                        "image_url": record.get("input", {}).get("image_url"),
                        "prediction": prediction_text,
                        "ground_truth": label_text,
                        "prediction_json": prediction_json,
                        "ground_truth_json": label_json,
                    }
                )

            predictions_file = os.path.join(paths.run_dir, "test_samples.json")
            with open(predictions_file, "w", encoding="utf-8") as f:
                json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
            logger.info(
                "Saved %d test sample predictions to %s",
                len(sample_predictions),
                predictions_file,
            )
        except Exception as exc:  # noqa: BLE001 - surface prediction issues
            logger.error("Error generating test samples: %s", exc)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

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
