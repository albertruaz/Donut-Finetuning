import json
import os
import time
from dataclasses import dataclass


@dataclass
class RunPaths:
    run_dir: str
    logs_dir: str
    model_dir: str
    ckpt_dir: str


def load_config(path: str = "config.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_run_dirs(output_root: str) -> RunPaths:
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, ts)
    logs_dir = os.path.join(run_dir, "logs")
    model_dir = os.path.join(run_dir, "model")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    for d in [run_dir, logs_dir, model_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)
    return RunPaths(run_dir, logs_dir, model_dir, ckpt_dir)

