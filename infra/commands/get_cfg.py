"""Show the effective Hydra config for train or test.

Examples:
  python get_cfg.py --train
  python get_cfg.py --test
"""

from __future__ import annotations

import argparse
import sys, os
from pathlib import Path
from typing import Any, Dict

from hydra import compose, initialize, initialize_config_dir
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _abs_path(repo_root: Path, rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else (repo_root / p).resolve())


def _abs_data_dir(repo_root: Path, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(dataset_cfg)
    cfg["data_dir"] = _abs_path(repo_root, cfg.get("data_dir", "./data"))
    return cfg


def _yaml_with_defaults(cfg) -> str:
    """Return YAML with resolved values and an optional __defaults__ section."""
    try:
        defaults_list = list(getattr(cfg, "defaults"))
    except Exception:
        defaults_list = []
    merged = OmegaConf.merge(OmegaConf.create({"__defaults__": defaults_list}), cfg)
    return OmegaConf.to_yaml(merged, resolve=True)


def show_train_cfg(expanded: bool = False) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config")
    if expanded:
        return _yaml_with_defaults(cfg)

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


def show_test_cfg(expanded: bool = False) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="test")
    if expanded:
        return _yaml_with_defaults(cfg)

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


def _try_load_yaml_direct(repo_root: Path, name: str):
    # If user passed a path, respect it; else search under configs/ with both extensions
    cand = Path(name)
    if cand.suffix in {".yaml", ".yml"} and cand.exists():
        return OmegaConf.load(str(cand))
    for ext in (".yaml", ".yml"):
        p = repo_root / "configs" / f"{name}{ext}"
        if p.exists():
            return OmegaConf.load(str(p))
    return None


def show_named_cfg(name: str, expanded: bool = False) -> str:
    """Compose and print any config by name; fallback to direct YAML load if needed."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg_dir = str((repo_root / "configs").resolve())
    try:
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name=name)
        return _yaml_with_defaults(cfg) if expanded else OmegaConf.to_yaml(cfg)
    except MissingConfigException:
        cfg = _try_load_yaml_direct(repo_root, name)
        if cfg is None:
            raise
        return OmegaConf.to_yaml(cfg)


def main():
    parser = argparse.ArgumentParser(description="Show effective configs for train/test or a named config")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--train", action="store_true", help="Show train configuration")
    group.add_argument("--test", action="store_true", help="Show test configuration")
    parser.add_argument(
        "config",
        nargs="?",
        help="Optional config name (without .yaml) under configs/ to compose and print",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Show fully composed config with resolved values and __defaults__",
    )

    args = parser.parse_args()

    # Support Hydra-style token: config=name
    if args.config and args.config.startswith("config="):
        args.config = args.config.split("=", 1)[1]

    if args.config:
        print(show_named_cfg(args.config, expanded=args.expanded))
    elif args.train:
        print(show_train_cfg(expanded=args.expanded))
    elif args.test:
        print(show_test_cfg(expanded=args.expanded))
    else:
        parser.error("Specify --train, --test, or a config name (e.g., train_ivae_mlp)")


if __name__ == "__main__":
    main()
