import argparse
import json
import os
import time
from typing import Any, List, Optional, Sequence, Tuple

from sklearn.model_selection import train_test_split


DEFAULT_INPUT_PATH = os.path.join("../data", "donut_finetune_dataset.jsonl")
DEFAULT_OUTPUTS = {
    "train": os.path.join("../data", "train_dataset.jsonl"),
    "valid": os.path.join("../data", "validation_dataset.jsonl"),
    "test": os.path.join("../data", "test_dataset.jsonl"),
}


def load_jsonl(path: str) -> List[Any]:
    records: List[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dump_jsonl(path: str, records: Sequence[Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def maybe_build_stratify_labels(
    records: Sequence[Any], cfg_split: dict
) -> Optional[List[Any]]:
    if not cfg_split.get("stratify"):
        return None

    key_path = cfg_split.get("stratify_key")
    if not key_path:
        print(
            "[WARN] stratify=true but split.stratify_key missing in config; proceeding without stratification."
        )
        return None

    keys = key_path.split(".")
    labels: List[Any] = []
    for record in records:
        value: Any = record
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        if value is None or (isinstance(value, float) and (value != value)):
            value = "__missing__"
        labels.append(value)

    # If all labels collapse to a single bucket, sklearn will complain → skip stratify
    if len(set(labels)) <= 1:
        print(
            "[WARN] stratify labels collapsed to a single value; proceeding without stratification."
        )
        return None
    return labels


def split_records(
    records: Sequence[Any],
    labels: Optional[Sequence[Any]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
    if not records:
        raise ValueError("No records found to split.")

    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    remainder_ratio = 1.0 - train_ratio

    if remainder_ratio <= 0:
        return list(records), [], []

    stratify_labels = labels if labels is not None else None

    if stratify_labels is not None:
        train_records, temp_records, _, temp_labels = train_test_split(
            records,
            stratify_labels,
            test_size=remainder_ratio,
            shuffle=shuffle,
            random_state=seed,
            stratify=stratify_labels,
        )
    else:
        train_records, temp_records = train_test_split(
            records,
            test_size=remainder_ratio,
            shuffle=shuffle,
            random_state=seed,
            stratify=None,
        )
        temp_labels = None

    if valid_ratio == 0 and test_ratio == 0:
        return train_records, [], []

    if valid_ratio == 0:
        return train_records, [], list(temp_records)

    if test_ratio == 0:
        return train_records, list(temp_records), []

    valid_ratio_rel = valid_ratio / (valid_ratio + test_ratio)

    if temp_labels is not None:
        valid_records, test_records, _, _ = train_test_split(
            temp_records,
            temp_labels,
            test_size=(1.0 - valid_ratio_rel),
            shuffle=shuffle,
            random_state=seed,
            stratify=temp_labels,
        )
    else:
        valid_records, test_records = train_test_split(
            temp_records,
            test_size=(1.0 - valid_ratio_rel),
            shuffle=shuffle,
            random_state=seed,
            stratify=None,
        )

    return list(train_records), list(valid_records), list(test_records)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    split_cfg = cfg.get("split", {})
    tr = float(split_cfg.get("train_ratio", 0.8))
    vr = float(split_cfg.get("valid_ratio", 0.1))
    te = float(split_cfg.get("test_ratio", 0.1))
    shuffle = bool(split_cfg.get("shuffle", True))
    seed = int(cfg.get("seed", 42))

    input_path = args.input or cfg.get("data", {}).get("processed_jsonl") or DEFAULT_INPUT_PATH
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    records = load_jsonl(input_path)
    stratify_labels = maybe_build_stratify_labels(records, split_cfg)

    train_records, valid_records, test_records = split_records(
        records, stratify_labels, tr, vr, te, shuffle, seed
    )

    outputs = DEFAULT_OUTPUTS.copy()
    if args.output_dir:
        outputs = {
            "train": os.path.join(args.output_dir, os.path.basename(outputs["train"])),
            "valid": os.path.join(args.output_dir, os.path.basename(outputs["valid"])),
            "test": os.path.join(args.output_dir, os.path.basename(outputs["test"])),
        }

    dump_jsonl(outputs["train"], train_records)
    dump_jsonl(outputs["valid"], valid_records)
    dump_jsonl(outputs["test"], test_records)

    summary_path = os.path.join(os.path.dirname(outputs["train"]), "data_split_summary.json")
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "source": os.path.abspath(input_path),
        "outputs": {k: os.path.abspath(v) for k, v in outputs.items()},
        "counts": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
        },
        "ratios": {"train": tr, "valid": vr, "test": te},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved splits:", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="divide_config.json")
    ap.add_argument(
        "--input",
        default=None,
        help="Path to the processed JSONL produced by transform_data.py (overrides config).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write the split JSONL files (defaults to data/).",
    )
    main(ap.parse_args())
