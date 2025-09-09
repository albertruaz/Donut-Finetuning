import argparse
import os
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    with open(args.config, "r") as f:
        cfg = json.load(f)

    src_csv = os.path.join("data", "original_data", "image_data.csv")
    df = pd.read_csv(src_csv)

    assert {"image_path", "text"} <= set(df.columns), \
        "CSV must have columns: image_path,text"

    tr = cfg["split"]["train_ratio"]
    vr = cfg["split"]["valid_ratio"]
    te = cfg["split"]["test_ratio"]
    assert abs((tr + vr + te) - 1.0) < 1e-6

    shuffle = cfg["split"]["shuffle"]
    stratify = df["text"] if cfg["split"]["stratify"] else None
    seed = cfg.get("seed", 42)

    df_train, df_temp = train_test_split(
        df, test_size=(1 - tr), shuffle=shuffle, random_state=seed, stratify=stratify
    )
    valid_ratio_rel = vr / (vr + te) if (vr + te) > 0 else 0
    df_valid, df_test = train_test_split(
        df_temp,
        test_size=(1 - valid_ratio_rel),
        shuffle=shuffle,
        random_state=seed,
        stratify=(df_temp["text"] if cfg["split"]["stratify"] else None),
    )

    os.makedirs("data", exist_ok=True)
    df_train.to_csv("data/train_data.csv", index=False)
    df_valid.to_csv("data/validation_data.csv", index=False)
    df_test.to_csv("data/test_data.csv", index=False)

    # simple log
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "counts": {"train": len(df_train), "valid": len(df_valid), "test": len(df_test)},
    }
    with open("data/data_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved splits:", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    main(args)

