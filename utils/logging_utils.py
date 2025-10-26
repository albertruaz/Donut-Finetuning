import logging
import os
import sys
from typing import Any, Dict, Optional

from utils.config_manager import RunPaths


def setup_logger(log_dir: str):
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(log_dir, "application.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    eh = logging.FileHandler(os.path.join(log_dir, "error.log"))
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def initialise_wandb(cfg: Dict[str, Any], paths: RunPaths, logger: logging.Logger):
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

    init_kwargs: Dict[str, Any] = dict(wandb_cfg.get("init_kwargs", {}))
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

    def _init_with_kwargs(kwargs: Dict[str, Any]):
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

