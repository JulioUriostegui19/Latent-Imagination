"""Show the effective Hydra config for train or test.

Examples:
  python get_cfg.py --train
  python get_cfg.py --test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from hydra import compose, initialize
from omegaconf import OmegaConf


def _abs_path(repo_root: Path, rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else (repo_root / p).resolve())


def _abs_data_dir(repo_root: Path, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(dataset_cfg)
    cfg["data_dir"] = _abs_path(repo_root, cfg.get("data_dir", "./data"))
    return cfg


def show_train_cfg() -> str:
    repo_root = Path(__file__).resolve().parents[2]  # repo root (../../ from infra/comants)
    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="config")
    finally:
        os.chdir(cwd)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    dataset_cfg = _abs_data_dir(repo_root, dataset_cfg)
    train_cfg["save_dir"] = _abs_path(repo_root, train_cfg.get("save_dir", "./runs"))

    effective = {
        "dataset": dataset_cfg,
        "model": model_cfg,
        "train": train_cfg,
    }
    return OmegaConf.to_yaml(OmegaConf.create(effective))


def show_test_cfg() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="test")
    finally:
        os.chdir(cwd)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    test_settings = OmegaConf.to_container(cfg.test_settings, resolve=True)
    output_dir = _abs_path(repo_root, cfg.get("output_dir", "./analytics"))

    dataset_cfg = _abs_data_dir(repo_root, dataset_cfg)

    effective = {
        "dataset": dataset_cfg,
        "test_settings": test_settings,
        "models": list(cfg.get("models", [])),
        "tests": list(cfg.get("tests", [])),
        "output_dir": output_dir,
    }
    return OmegaConf.to_yaml(OmegaConf.create(effective))


def main():
    parser = argparse.ArgumentParser(description="Show effective configs for train/test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Show train configuration")
    group.add_argument("--test", action="store_true", help="Show test configuration")

    args = parser.parse_args()

    if args.train:
        print(show_train_cfg())
    else:
        print(show_test_cfg())


if __name__ == "__main__":
    main()
