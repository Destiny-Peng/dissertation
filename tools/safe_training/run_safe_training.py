"""Run the official SAFE Hydra trainer with LF3R project paths."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", choices=("mlp", "lstm"), required=True, help="mlp maps to upstream model=indep")
    parser.add_argument("--gpu", required=True, help="CUDA device visible to this process")
    parser.add_argument("--logs-root", default="outputs/safe_training/logs")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", default="0")
    parser.add_argument("--token-idx-rel", default="1.0")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--extra", action="append", default=[], help="Additional Hydra override; repeatable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    safe_repo = root / "repos" / "SAFE"
    safe_python = Path(os.environ.get("LF3R_SAFE_PYTHON", root / "conda_envs" / "LF3R-safe" / "bin" / "python"))
    dataset = Path(args.dataset_dir).expanduser()
    if not dataset.is_absolute():
        dataset = root / dataset
    dataset = dataset.resolve()
    logs = Path(args.logs_root).expanduser()
    if not logs.is_absolute():
        logs = root / logs
    logs = logs.resolve()
    for path, name in ((safe_repo, "SAFE repository"), (dataset, "dataset"), (logs, "logs")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SystemExit(f"{name} must stay inside project root: {path}") from exc
    if not safe_python.is_file():
        raise SystemExit(f"SAFE interpreter missing: {safe_python}")
    if not dataset.is_dir():
        raise SystemExit(f"SAFE dataset directory missing: {dataset}")
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1:
        raise SystemExit("epochs, batch-size, and hidden-dim must be positive")
    model = "indep" if args.model == "mlp" else "lstm"
    command = [
        str(safe_python), "-m", "failure_prob.train",
        "dataset=openvla", f"model={model}", "dataset.data_path_prefix=",
        f"dataset.data_path={dataset.as_posix()}/", "dataset.load_to_cuda=true",
        f"dataset.token_idx_rel={args.token_idx_rel}",
        f"dataset.normalize_hidden_states={'true' if args.normalize else 'false'}",
        f"model.n_epochs={args.epochs}", f"model.batch_size={args.batch_size}",
        f"model.hidden_dim={args.hidden_dim}", f"train.seed={args.seed}",
        "train.eval_save_ckpt=true", f"train.logs_save_root={logs.as_posix()}",
        f"train.wandb_dir={(logs / 'wandb').as_posix()}",
    ]
    command.extend(args.extra)
    print("Official command:")
    print(" ".join(shlex.quote(value) for value in command))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("WANDB_MODE", "offline")
    return subprocess.run(command, cwd=safe_repo, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
