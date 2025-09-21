import argparse
import json
import os
from typing import Any, Dict, Optional

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
from utils.config_manager import load_config, make_run_dirs
from utils.logging_utils import setup_logger


def _safe_json_loads(text: str) -> Any:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _strip_prompt(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :]
    return text


def build_compute_metrics(processor, prompt: str):
    pad_token_id = processor.tokenizer.pad_token_id

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [_strip_prompt(pred.strip(), prompt) for pred in decoded_preds]
        decoded_labels = [_strip_prompt(label.strip(), prompt) for label in decoded_labels]

        pred_objs = [_safe_json_loads(pred) for pred in decoded_preds]
        label_objs = [_safe_json_loads(label) for label in decoded_labels]

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
                    if pred.get(key, "") == gold.get(key, ""):
                        field_correct[key] = field_correct.get(key, 0) + 1
            else:
                if pred == gold:
                    exact += 1

        metrics: Dict[str, float] = {}
        metrics["exact_match"] = exact / total if total else 0.0
        for key in field_totals:
            metrics[f"acc_{key}"] = field_correct.get(key, 0) / field_totals[key]
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json", help="Path to config JSON")
    return ap.parse_args()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = make_run_dirs(cfg["output_root"])
    logger = setup_logger(paths.logs_dir)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info("Seed fixed to %d", seed)

    model_name = cfg["model_name"]
    processor = DonutProcessor.from_pretrained(model_name)
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

    collator = DonutDataCollator(processor)

    train_cfg = cfg.get("train", {})
    generation_cfg = cfg.get("generation", {})
    metric_for_best_model = train_cfg.get("metric_for_best_model", "exact_match")

    training_args = Seq2SeqTrainingArguments(
        output_dir=paths.ckpt_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        learning_rate=train_cfg.get("learning_rate", 5e-5),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "steps")
        if valid_dataset is not None
        else "no",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        fp16=train_cfg.get("fp16", False),
        report_to=train_cfg.get("report_to", []),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        logging_dir=paths.logs_dir,
        seed=seed,
        predict_with_generate=True,
        generation_max_length=generation_cfg.get("max_length", max_target_length),
        generation_num_beams=generation_cfg.get("num_beams", 1),
        remove_unused_columns=False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=train_cfg.get("greater_is_better", True),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=build_compute_metrics(processor, task_prompt)
        if valid_dataset is not None
        else None,
    )

    logger.info("Starting training for %s", model_name)
    trainer.train()

    logger.info("Saving model and processor to %s", paths.model_dir)
    model.save_pretrained(paths.model_dir)
    processor.save_pretrained(paths.model_dir)

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
