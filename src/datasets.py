import json
import math
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # NaN 값을 null로 치환하여 JSON 파싱 오류 방지
                line = line.replace(": NaN", ": null").replace(":NaN", ":null")
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON line: {line[:100]}... Error: {e}")
                continue
    return records


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _gt_to_token_string(data: Dict[str, Any]) -> str:
    """Converts ground truth dict to a token-based string."""
    parts = []
    for key, value in sorted(data.items()):
        if value is None or _is_nan(value):
            value = ""
        # str()로 명시적 변환
        value_str = str(value).strip()
        parts.append(f"<s_{key}>{value_str}</s_{key}>")
    return "".join(parts)


class DonutJsonlDataset(Dataset):
    """Dataset wrapping Donut-style JSONL records.

    Each JSONL row is expected to have the following structure::

        {
            "input": {
                "image_path": "path/to/img.png" | null,
                "image_url": "https://..." | null,
                "rotate_to": 0 | 1 | 2 | 3 (clockwise quarter turns)
            },
            "output": {
                "brand": "...",
                "size": "...",
                "material": "..."
            }
        }

    "output" can contain any mapping of key -> value. Missing values (None / NaN)
    are converted to empty strings before serialisation. The final training target is::

        task_prompt + _gt_to_token_string(output_dict) + eos_token
    """

    def __init__(
        self,
        jsonl_path: str,
        processor,
        task_prompt: str = "",
        max_target_length: int = 512,
        image_root: Optional[str] = None,
        image_cache_dir: Optional[str] = None,
        download_timeout: int = 10,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.task_prompt = task_prompt or ""
        self.max_target_length = max_target_length
        self.image_root = Path(image_root) if image_root else None
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        self.download_timeout = download_timeout
        self._session: Optional[requests.Session] = None

        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        self.records = _read_jsonl(jsonl_path)
        if len(self.records) == 0:
            raise ValueError(f"JSONL file {jsonl_path} is empty")
        
        # Ground truth를 token string으로 변환하여 캐싱
        self.processed_records = []
        for record in self.records:
            output_data = record.get("output", {})
            ground_truth = _gt_to_token_string(output_data)
            self.processed_records.append(
                {
                    "input": record.get("input", {}),
                    "ground_truth_str": ground_truth,
                    "original_output": output_data, # for validation
                }
            )

        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Lazily align pad token with EOS if missing
            eos_token = self.processor.tokenizer.eos_token
            if eos_token is None:
                raise ValueError("Tokenizer must define either pad_token or eos_token")
            self.processor.tokenizer.pad_token = eos_token
            self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.processed_records)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _ensure_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _cache_remote_image(self, url: str) -> Path:
        if self.image_cache_dir is None:
            raise ValueError(
                "Remote image encountered but image_cache_dir is not configured."
            )
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(url)
        extension = Path(parsed.path).suffix or ".jpg"
        cache_name = hashlib.md5(url.encode("utf-8")).hexdigest() + extension
        cache_path = self.image_cache_dir / cache_name

        if not cache_path.exists():
            resp = self._ensure_session().get(url, timeout=self.download_timeout)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        return cache_path

    def _load_image(self, input_payload: Dict[str, Any]) -> Image.Image:
        image_path = input_payload.get("image_path")
        image_url = input_payload.get("image_url")

        if image_path:
            path = Path(image_path)
            if not path.is_absolute() and self.image_root:
                path = self.image_root / path
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
        elif image_url:
            path = self._cache_remote_image(str(image_url))
        else:
            raise ValueError("Record must contain either input.image_path or input.image_url")

        image = Image.open(path).convert("RGB")

        rotate_to = input_payload.get("rotate_to")
        if rotate_to:
            try:
                quarter_turns = int(round(float(rotate_to))) % 4
            except (TypeError, ValueError):
                quarter_turns = 0
            if quarter_turns:
                # PIL rotates counter-clockwise for positive angles
                image = image.rotate(-90 * quarter_turns, expand=True)
        return image

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.processed_records[idx]
        input_data = record.get("input", {})
        image = self._load_image(input_data)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        target_sequence = self.task_prompt + record["ground_truth_str"]
        tokenized = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=True,  # EOS is added
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

        # For seq2seq, labels are the same as input_ids, but with padding masked
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100  # Mask padding

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class DonutDataCollator:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            eos_token = processor.tokenizer.eos_token
            if eos_token is None:
                raise ValueError("Tokenizer must define either pad_token or eos_token")
            processor.tokenizer.pad_token = eos_token
            self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        labels = [example["labels"] for example in batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels_padded[labels_padded == self.pad_token_id] = -100
        return {
            "pixel_values": pixel_values,
            "labels": labels_padded,
        }
