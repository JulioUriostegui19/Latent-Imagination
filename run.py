"""Single entrypoint to run project commands with correct PYTHONPATH.

Usage examples:
  python run.py train train.epochs=1
  python run.py test models=["path.ckpt"] tests=["iterative","ood"]
  python run.py get_cfg --train
  python run.py dashboard --launch-tensorboard --logdir runs
  python run.py get_model --tools torchinfo --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Project runner")
    parser.add_argument(
        "command",
        choices=["train", "test", "get_cfg", "dashboard", "get_model"],
        help="Subcommand to run",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Args to pass through")
    opts = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])

    script = repo_root / "infra" / "commands" / f"{opts.command}.py"
    if not script.exists():
        print(f"Unknown command or missing script: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script), *opts.args]
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

