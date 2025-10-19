"""Colab-friendly dashboard launcher for this VAE project.

Goals:
- Launch TensorBoard for live tracking (non-blocking, background).
- Print effective train/test configs.
- Quick model overview via torchinfo/hiddenlayer/TensorBoard graph.
- Simple hyperparameter trial runner (sequential), suitable for quick tests.

Example Colab workflow:
  !python dashboard.py --launch-tensorboard --logdir runs --port 6006
  !python train.py               # training logs stream into TensorBoard

Additional examples:
  # Show composed train config
  !python dashboard.py --print-cfg train

  # Model overview (MNIST-like encoder)
  !python dashboard.py --model-overview \
      --tools torchinfo tensorboard \
      --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32

  # Run a couple of quick trials (sequential)
  !python dashboard.py --run-sweep "train.epochs=2 model.beta=0.5::train.epochs=2 model.beta=2.0"
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
from itertools import product
from pathlib import Path
from typing import List


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def launch_tensorboard(
    logdir: str, port: int, host: str, background: bool = True
) -> str:
    """Start TensorBoard pointing at `logdir`. Returns human-friendly info string.

    In Colab, this runs as a background process so you can start training next.
    """
    _ensure_dir(logdir)
    cmd = [
        "tensorboard",
        f"--logdir={logdir}",
        f"--port={port}",
        f"--host={host}",
    ]

    if background:
        # Detach process group so it survives this script's exit.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
        )
        pid_file = Path(logdir) / ".tensorboard_pid"
        pid_file.write_text(str(proc.pid))
        url = f"http://{host}:{port}"
        return f"TensorBoard started (PID {proc.pid}). Open {url}"
    else:
        # Foreground (blocking). Useful if you want to tail logs.
        print("Running:", " ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)
        return "TensorBoard exited"


def stop_tensorboard(logdir: str) -> str:
    pid_path = Path(logdir) / ".tensorboard_pid"
    if not pid_path.exists():
        return f"No PID file found at {pid_path}"
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        return f"Invalid PID file: {pid_path}"

    try:
        # Try killing the whole process group if we created one.
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            return f"Failed to stop TensorBoard (PID {pid}): {exc}"
    try:
        pid_path.unlink()
    except Exception:
        pass
    return f"Stopped TensorBoard (PID {pid})"


def print_cfg(which: str) -> None:
    if which == "train":
        from get_cfg import show_train_cfg

        print(show_train_cfg())
    elif which == "test":
        from get_cfg import show_test_cfg

        print(show_test_cfg())
    else:
        raise ValueError("--print-cfg must be 'train' or 'test'")


def model_overview(args) -> None:
    try:
        from infra.comants.get_model import get_mode
    except Exception as exc:
        print(f"Model overview unavailable (import failed): {exc}")
        return

    results = get_mode(
        tools=args.tools,
        model_type=args.model_type,
        input_shape=args.input_shape,
        hidden_dims=args.hidden_dims,
        z_dim=args.z_dim,
        visual_batch_size=args.visual_batch_size,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )
    print("=== model overview ===")
    for k, v in results.items():
        print(f"[{k}] {v}")


def run_sweep(spec: str, base_overrides: List[str] | None = None) -> None:
    """Run a simple sequential sweep.

    spec format: a '::'-separated list of override strings. Each override string
    is space-separated like: "train.epochs=2 model.beta=0.5".
    """
    base_overrides = base_overrides or []
    combos = [s.strip() for s in spec.split("::") if s.strip()]
    if not combos:
        print("No valid overrides in --run-sweep spec")
        return
    for i, combo in enumerate(combos, 1):
        overrides = base_overrides + combo.split()
        cmd = ["python", "train.py", *overrides]
        print(f"\n[trial {i}/{len(combos)}]", " ".join(shlex.quote(x) for x in cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Trial failed with exit code {exc.returncode}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project dashboard helpers")
    p.add_argument(
        "--launch-tensorboard",
        action="store_true",
        help="Start TensorBoard in background",
    )
    p.add_argument(
        "--stop-tensorboard", action="store_true", help="Stop background TensorBoard"
    )
    p.add_argument(
        "--logdir", default="runs", help="TensorBoard logdir (default: runs)"
    )
    p.add_argument(
        "--port", type=int, default=6006, help="TensorBoard port (default: 6006)"
    )
    p.add_argument(
        "--host", default="0.0.0.0", help="TensorBoard host (default: 0.0.0.0)"
    )

    p.add_argument(
        "--print-cfg", choices=["train", "test"], help="Print effective Hydra config"
    )

    p.add_argument(
        "--model-overview", action="store_true", help="Run quick model overview tools"
    )
    p.add_argument(
        "--tools",
        nargs="+",
        default=["torchinfo"],
        help="Tools: torchinfo tensorboard hiddenlayer",
    )
    p.add_argument("--model-type", choices=["encoder", "decoder"], default="encoder")
    p.add_argument(
        "--input-shape", nargs=3, type=int, metavar=("C", "H", "W"), default=[1, 28, 28]
    )
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 64])
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--visual-batch-size", type=int, default=1)
    p.add_argument("--tensorboard-log-dir", type=str)

    p.add_argument(
        "--run-sweep",
        type=str,
        help=(
            "Sequential trials spec; '::'-separated override strings.\n"
            'Example: "train.epochs=2 model.beta=0.5::train.epochs=2 model.beta=2.0"'
        ),
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.launch_tensorboard:
        msg = launch_tensorboard(args.logdir, args.port, args.host, background=True)
        print(msg)

    if args.stop_tensorboard:
        print(stop_tensorboard(args.logdir))

    if args.print_cfg:
        print_cfg(args.print_cfg)

    if args.model_overview:
        model_overview(args)

    if args.run_sweep:
        run_sweep(args.run_sweep)

    # If nothing selected, show help
    if not any(
        [
            args.launch_tensorboard,
            args.stop_tensorboard,
            args.print_cfg is not None,
            args.model_overview,
            args.run_sweep,
        ]
    ):
        print("No action selected. See --help for options.")


if __name__ == "__main__":
    main()
