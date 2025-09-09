import os
import json
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments,
)
from utils.config_manager import load_config, make_run_dirs
from utils.logging_utils import setup_logger
from src.datasets import DonutCsvDataset, DonutDataCollator


def main():
    cfg = load_config("config.json")
    paths = make_run_dirs(cfg["output_root"])
    logger = setup_logger(paths.logs_dir)
    logger.info("Config loaded.")

    model_name = cfg["model_name"]
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # decoder/pad settings (needed for training and generation)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
        )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Datasets
    train_ds = DonutCsvDataset(
        cfg["data"]["train_csv"], processor, cfg["data"]["image_root"]
    )
    valid_ds = DonutCsvDataset(
        cfg["data"]["valid_csv"], processor, cfg["data"]["image_root"]
    )
    collator = DonutDataCollator(processor)

    # Training args
    ta = TrainingArguments(
        output_dir=paths.ckpt_dir,
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        learning_rate=cfg["train"]["learning_rate"],
        warmup_steps=cfg["train"]["warmup_steps"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=cfg["train"]["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["train"]["save_steps"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        load_best_model_at_end=True,
        fp16=cfg["train"]["fp16"],
        report_to=[],
        save_total_limit=2,
        logging_dir=paths.logs_dir,
        seed=cfg.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )

    logger.info("Start training...")
    trainer.train()

    # Save: model + processor + metadata
    model.save_pretrained(paths.model_dir)
    processor.save_pretrained(paths.model_dir)
    meta = {
        "config": cfg,
        "best_metric": getattr(trainer, "state", None)
        and trainer.state.best_metric,
        "global_step": getattr(trainer, "state", None)
        and trainer.state.global_step,
        "model_dir": paths.model_dir,
    }
    with open(os.path.join(paths.run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved to {paths.run_dir}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

