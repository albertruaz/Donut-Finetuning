import argparse
import io
import json
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from utils.config_manager import load_config


def _load_image(path_or_url: str) -> Image.Image:
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


@torch.inference_mode()
def run_inference(
    image_source: str,
    model_dir: str,
    task_prompt: str = "",
    generation_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    processor = DonutProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
        )

    image = _load_image(image_source)
    pixel_values = processor(image, return_tensors="pt").pixel_values

    prompt_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids

    gen_config = {
        "max_length": generation_cfg.get("max_length")
        if generation_cfg
        else getattr(model.config, "max_length", 512),
        "num_beams": generation_cfg.get("num_beams", 1)
        if generation_cfg
        else 1,
        "min_new_tokens": generation_cfg.get("min_new_tokens", 1)
        if generation_cfg
        else 1,
        "repetition_penalty": generation_cfg.get("repetition_penalty", 1.0)
        if generation_cfg
        else 1.0,
    }

    # 안전: generation_config에만 설정하고, generate() 호출은 심플하게 유지
    model.generation_config.max_length = gen_config["max_length"]
    model.generation_config.num_beams = gen_config["num_beams"]
    model.generation_config.min_new_tokens = gen_config["min_new_tokens"]
    model.generation_config.repetition_penalty = gen_config["repetition_penalty"]
    
    # (옵션) 지원될 때만 설정
    if hasattr(model.generation_config, "no_repeat_ngram_size"):
        model.generation_config.no_repeat_ngram_size = 3
    if hasattr(model.generation_config, "renormalize_logits"):
        setattr(model.generation_config, "renormalize_logits", True)

    generated = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=prompt_ids,
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    if task_prompt and sequence.startswith(task_prompt):
        sequence = sequence[len(task_prompt) :]
    return sequence.strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing the fine-tuned Donut model (results/run_*/model)",
    )
    ap.add_argument("--image", required=True, help="Path or URL to the target image")
    ap.add_argument(
        "--config",
        default=None,
        help="Optional config JSON to reuse task prompts and generation settings",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_prompt = ""
    generation = None
    if args.config:
        cfg = load_config(args.config)
        task_prompt = cfg.get("tokenizer", {}).get("task_prompt", "")
        generation = cfg.get("generation", {})

    result = run_inference(
        args.image,
        args.model_dir,
        task_prompt=task_prompt,
        generation_cfg=generation,
    )
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(result)
