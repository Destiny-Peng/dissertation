"""Load an official SAFE checkpoint and write finite validation scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("mlp", "lstm"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dataset_path = (root / args.dataset_dir if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)).resolve()
    checkpoint = (root / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)).resolve()
    output = (root / args.output if not Path(args.output).is_absolute() else Path(args.output)).resolve()
    for path in (dataset_path, checkpoint, output.parent):
        path.relative_to(root)
    if not dataset_path.is_dir() or not checkpoint.is_file():
        raise SystemExit("dataset or checkpoint is missing")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import sys
    sys.path.insert(0, str(root / "repos" / "SAFE"))
    from failure_prob.conf import Config, IndepModelConfig, LstmModelConfig, OpenvlaDatasetConfig, TrainConfig
    from failure_prob.data.openvla import load_rollouts, split_rollouts
    from failure_prob.data.utils import RolloutDataset
    from failure_prob.model import get_model
    cfg = Config(
        dataset=OpenvlaDatasetConfig(data_path=str(dataset_path) + "/", data_path_prefix="", load_to_cuda=True, unseen_task_ratio=0.25, seen_train_ratio=0.5, token_idx_rel=1.0),
        model=(IndepModelConfig(batch_size=args.batch_size, hidden_dim=args.hidden_dim, n_layers=2, lambda_reg=0.0) if args.model == "mlp" else LstmModelConfig(batch_size=args.batch_size, hidden_dim=args.hidden_dim, n_layers=1, lambda_reg=0.0)),
        train=TrainConfig(seed="0"),
    )
    rollouts = load_rollouts(cfg)
    splits = split_rollouts(cfg, rollouts)
    model = get_model(cfg, int(rollouts[0].hidden_states.shape[-1])).to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval()
    rows = []
    with torch.no_grad():
        for split_name, rollout_rows in splits.items():
            loader = DataLoader(RolloutDataset(cfg, rollout_rows, device="cuda"), batch_size=args.batch_size, shuffle=False, num_workers=0)
            offset = 0
            for batch in loader:
                scores = model(batch).squeeze(-1)
                for index in range(scores.shape[0]):
                    length = int(batch["valid_masks"][index].sum().item())
                    values = scores[index, :length].detach().float().cpu().tolist()
                    if not all(torch.isfinite(torch.tensor(values)).tolist()):
                        raise RuntimeError("non-finite validation score")
                    rollout = rollout_rows[offset + index]
                    rows.append({"split": split_name, "task_id": int(rollout.task_id), "episode_idx": int(rollout.episode_idx), "episode_success": int(rollout.episode_success), "scores": values})
                offset += scores.shape[0]
    payload = {"model": args.model, "checkpoint": str(checkpoint), "gpu": args.gpu, "rollouts": len(rows), "finite": True, "scores": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rollouts": len(rows), "finite": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
