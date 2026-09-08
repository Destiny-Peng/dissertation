"""Diagnose SAFE checkpoint separability on the exact official training split.

This LF3R-owned wrapper loads the official SAFE OpenVLA rollouts, applies the
same token selection used during training, scores every training rollout with
both saved checkpoints, and writes raw scores plus diagnostics. It never runs
LIBERO/OpenVLA or conformal calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "outputs/openvla_libero/lf3r-data-natural-libero10-20260906_084450/libero_10"
DEFAULT_OUTPUT = ROOT / "outputs/safe_training/diagnostics/lf3r_natural_libero10_train"
DEFAULT_MLP = ROOT / "outputs/safe_training/logs/lf3r_natural_libero10_20260906_mlp/openvla-default-indep-debug/20260907/173229/model_final.ckpt"
DEFAULT_LSTM = ROOT / "outputs/safe_training/logs/lf3r_natural_libero10_20260906_lstm/openvla-default-lstm-debug/20260907/174350/model_final.ckpt"


def encode(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=encode) + "\n", encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def make_config(dataset: Path, model_name: str):
    sys.path.insert(0, str(ROOT / "repos" / "SAFE"))
    from failure_prob.conf import Config, IndepModelConfig, LstmModelConfig, OpenvlaDatasetConfig, TrainConfig

    dataset_cfg = OpenvlaDatasetConfig(
        data_path=f"{dataset.as_posix()}/",
        data_path_prefix="",
        load_to_cuda=False,
        normalize_hidden_states=False,
        unseen_task_ratio=0.3,
        seen_train_ratio=0.6,
        token_idx_rel=1.0,
    )
    model_cfg = IndepModelConfig(hidden_dim=256, n_layers=2) if model_name == "mlp" else LstmModelConfig(hidden_dim=256, n_layers=1)
    return Config(dataset=dataset_cfg, model=model_cfg, train=TrainConfig(seed="0"))


def load_exact_split(dataset: Path):
    from failure_prob.data.openvla import load_rollouts, split_rollouts
    from failure_prob.utils.random import seed_everything

    cfg = make_config(dataset, "mlp")
    # Identical to failure_prob.train: seed before loading and again before split.
    seed_everything(0)
    all_rollouts = load_rollouts(cfg)
    seed_everything(0)
    splits = split_rollouts(cfg, all_rollouts)
    if "train" not in splits:
        raise RuntimeError("official split did not produce a train split")
    return cfg, all_rollouts, splits


def validate_rollouts(rollouts: list[Any]) -> dict[str, Any]:
    pattern = re.compile(r"--succ([01])(?:\.mp4)?$")
    lengths = []
    dimensions = set()
    labels = set()
    action_alignment = True
    filename_labels = True
    for rollout in rollouts:
        hidden = rollout.hidden_states
        dimensions.add(tuple(int(x) for x in hidden.shape))
        if hidden.ndim != 2 or hidden.shape[-1] != 4096:
            raise ValueError(f"expected selected hidden states (T,4096), got {tuple(hidden.shape)}")
        if not bool(torch.isfinite(hidden).all().item()):
            raise ValueError("non-finite hidden state")
        length = int(hidden.shape[0])
        lengths.append(length)
        labels.add(int(rollout.episode_success))
        if rollout.action_vectors is not None:
            action_alignment = action_alignment and len(rollout.action_vectors) == length
        match = pattern.search(str(rollout.mp4_path))
        if match is not None:
            filename_labels = filename_labels and int(match.group(1)) == int(rollout.episode_success)
    if labels != {0, 1} or not action_alignment or not filename_labels:
        raise ValueError("rollout preprocessing or label validation failed")
    return {
        "rollout_count": len(rollouts),
        "selected_feature_dimensions": sorted([list(x) for x in dimensions]),
        "input_dim": 4096,
        "token_idx_rel": 1.0,
        "selected_token_idx": 6,
        "hidden_states_finite": True,
        "native_action_alignment": action_alignment,
        "filename_label_mapping": filename_labels,
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "mean_length": float(np.mean(lengths)),
        "success_count": int(sum(int(r.episode_success) for r in rollouts)),
        "failure_count": int(sum(1 - int(r.episode_success) for r in rollouts)),
    }


def load_model(cfg: Any, checkpoint: Path, input_dim: int, device: torch.device):
    from failure_prob.model import get_model

    model = get_model(cfg, input_dim).to(device)
    state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def score_rollouts(model: Any, rollouts: list[Any], batch_size: int, device: torch.device):
    from failure_prob.data.utils import pad_rollout_batch

    outputs: list[np.ndarray] = []
    reset_verified = False
    with torch.inference_mode():
        for start in range(0, len(rollouts), batch_size):
            chunk = rollouts[start : start + batch_size]
            features, _masks, _labels, _actions = pad_rollout_batch(chunk, device="cpu")
            batch_features = features.to(device)
            scores = model({"features": batch_features}).squeeze(-1)
            if scores.ndim != 2 or scores.shape[0] != len(chunk):
                raise ValueError(f"unexpected score shape {tuple(scores.shape)}")
            if not bool(torch.isfinite(scores).all().item()):
                raise ValueError("non-finite detector score")
            for index, rollout in enumerate(chunk):
                length = int(rollout.hidden_states.shape[0])
                values = scores[index, :length].detach().float().cpu().numpy()
                if values.shape != (length,) or not np.isfinite(values).all():
                    raise ValueError("score/native timestep alignment or finiteness failed")
                outputs.append(values)
            # Official LSTM creates zero initial state on each call. Compare
            # batched and isolated first-rollout inference to verify reset.
            if not reset_verified:
                one_features, _, _, _ = pad_rollout_batch(chunk[:1], device="cpu")
                one_scores = model({"features": one_features.to(device)}).squeeze(-1)[0]
                length = int(chunk[0].hidden_states.shape[0])
                # cuDNN can use slightly different kernels for padded-batch and singleton LSTM shapes; allow only numerical noise while still checking valid native timesteps.
                reset_verified = bool(torch.allclose(one_scores[:length], scores[0, :length], atol=1e-2, rtol=1e-3))
                if not reset_verified:
                    raise RuntimeError("singleton/batched scores differ; sequence state may not reset")
    return outputs, reset_verified


def score_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("statistics require finite values")
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(median),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "min": float(values.min()), "max": float(values.max())}


def ordering(failures: np.ndarray, successes: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(failures)[:, None] - np.asarray(successes)[None, :]
    strict = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    total = int(diff.size)
    return {
        "failure_gt_success_pairs": strict,
        "ties": ties,
        "total_pairs": total,
        "failure_gt_success_fraction": float(strict / total),
        "failures_gt_success_median": int((np.asarray(failures) > np.median(successes)).sum()),
        "failure_count": int(len(failures)),
        "failures_gt_all_success": int((np.asarray(failures) > np.max(successes)).sum()),
    }


def summarize(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    labels = np.asarray([r["failure_label"] for r in records], dtype=np.int64)
    metrics = {
        "final": np.asarray([r[model_name]["final"] for r in records], dtype=float),
        "max": np.asarray([r[model_name]["max"] for r in records], dtype=float),
        "mean": np.asarray([r[model_name]["mean"] for r in records], dtype=float),
    }
    if model_name == "mlp":
        metrics["final_per_step"] = np.asarray([r[model_name]["final_per_step"] for r in records], dtype=float)
    from sklearn.metrics import average_precision_score, roc_auc_score

    result = {}
    for metric, values in metrics.items():
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite summary: {model_name}/{metric}")
        success = values[labels == 0]
        failure = values[labels == 1]
        result[metric] = {
            "success": score_stats(success),
            "failure": score_stats(failure),
            "auroc_failure_positive": float(roc_auc_score(labels, values)),
            "auprc_failure_positive": float(average_precision_score(labels, values)),
            "ordering": ordering(failure, success),
        }
    return result


def choose_representatives(records: list[dict[str, Any]], model_name: str):
    metric = "final_per_step" if model_name == "mlp" else "max"
    for role, label in (("clean_success", 0), ("clear_failure", 1)):
        candidates = [i for i, row in enumerate(records) if row["failure_label"] == label]
        values = np.asarray([records[i][model_name][metric] for i in candidates], dtype=float)
        target = float(np.median(values)) if role == "clean_success" else float(values.max())
        index = candidates[int(np.argmin(np.abs(values - target)))]
        row = records[index]
        yield role, {
            "role": role,
            "selection_metric": metric,
            "record_index": index,
            "task_id": row["task_id"],
            "episode_idx": row["episode_idx"],
            "episode_success": row["episode_success"],
            "length": row["length"],
            "task_description": row["task_description"],
            "selection_value": float(row[model_name][metric]),
            "scores": row[model_name]["scores"],
        }


def make_plots(output: Path, records: list[dict[str, Any]], representatives: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.asarray([r["failure_label"] for r in records], dtype=np.int64)
    panels = [
        ("MLP normalized final", "mlp", "final_per_step"),
        ("MLP final cumulative", "mlp", "final"),
        ("LSTM maximum", "lstm", "max"),
        ("LSTM mean", "lstm", "mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = ["#4472c4", "#c0504d"]
    for ax, (title, model_name, metric) in zip(axes.flat, panels):
        values = np.asarray([r[model_name][metric] for r in records], dtype=float)
        ax.boxplot([values[labels == 0], values[labels == 1]], tick_labels=["success", "failure"], showmeans=True)
        jitter = np.random.default_rng(0).normal(0, 0.035, size=values.size)
        ax.scatter(1 + jitter[labels == 0], values[labels == 0], s=10, alpha=0.35, color=colors[0])
        ax.scatter(2 + jitter[labels == 1], values[labels == 1], s=10, alpha=0.35, color=colors[1])
        ax.set_title(title)
        ax.set_ylabel("rollout-level score")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SAFE train-set score distributions (failure-positive direction)")
    fig.savefig(output / "score_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, model_name, title, ylabel in (
        (axes[0], "mlp", "SAFE-MLP representative curves", "cumulative score"),
        (axes[1], "lstm", "SAFE-LSTM representative curves", "failure probability"),
    ):
        for role, color in (("clean_success", "#4472c4"), ("clear_failure", "#c0504d")):
            row = representatives[model_name][role]
            x = np.arange(1, len(row["scores"]) + 1)
            ax.plot(x, row["scores"], color=color, linewidth=1.2,
                    label=f"{role} (task {row['task_id']}, ep {row['episode_idx']})")
        ax.set_title(title)
        ax.set_xlabel("native timestep")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(output / "representative_curves.png", dpi=180)
    plt.close(fig)


def make_report(output: Path, metadata: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    lines = [
        "# SAFE train-set separability diagnostic",
        "",
        "Training-set diagnostic only: no LIBERO/OpenVLA rollout and no conformal calibration.",
        "",
        "## Scope and verification",
        "",
        f"- Exact official split: seed 0, unseen_task_ratio=0.3, seen_train_ratio=0.6; train={metadata['train_rollouts']} rollouts, seen tasks {metadata['seen_task_ids']}.",
        f"- Labels: episode_success=1 is success; diagnostic positive class is failure=1-episode_success ({metadata['success_rollouts']} success, {metadata['failure_rollouts']} failure).",
        "- Preprocessing: official SAFE loader, source (T,7,4096), token_idx_rel=1.0 selecting token index 6, native (T,4096), no normalization.",
        "- Checkpoints loaded with strict state-dict matching; selected features, scores, and summaries are finite; score lengths equal native rollout lengths.",
        "- LSTM reset check: singleton and batched inference agree on the first training rollout, confirming fresh sequence state per rollout.",
        "",
        "## Rollout-level diagnostics",
        "",
        "Larger scores are treated as more failure-like. AUROC/AUPRC are descriptive train-set numbers, not test performance.",
        "",
    ]
    for model_name, title in (("mlp", "### SAFE-MLP"), ("lstm", "### SAFE-LSTM")):
        lines.extend([title, "", "| summary | success mean / median / IQR / min / max | failure mean / median / IQR / min / max | AUROC | AUPRC |", "|---|---:|---:|---:|---:|"])
        for metric, values in diagnostics[model_name].items():
            s, f = values["success"], values["failure"]
            lines.append(f"| {metric} | {s['mean']:.6g} / {s['median']:.6g} / {s['iqr']:.6g} / {s['min']:.6g} / {s['max']:.6g} | {f['mean']:.6g} / {f['median']:.6g} / {f['iqr']:.6g} / {f['min']:.6g} / {f['max']:.6g} | {values['auroc_failure_positive']:.4f} | {values['auprc_failure_positive']:.4f} |")
        lines.append("")
    lines.extend(["## Failure-over-success ordering", "",
                  "Counts below are strict pairwise comparisons across all failure x success rollout pairs.",
                  "", "| model / summary | failure > success pairs | total pairs | fraction | failures > all successes |", "|---|---:|---:|---:|---:|"])
    for model_name, values in diagnostics.items():
        for metric, item in values.items():
            order = item["ordering"]
            lines.append(f"| {model_name}/{metric} | {order['failure_gt_success_pairs']} | {order['total_pairs']} | {order['failure_gt_success_fraction']:.4f} | {order['failures_gt_all_success']} |")
    lines.extend(["", "## Interpretation", "",
                   "Use the accompanying plots to inspect overlap and native-time curves. This diagnostic intentionally does not choose a threshold or proceed to conformal calibration.",
                   "", "Static plots: score_distributions.png and representative_curves.png. Raw scores: raw_scores.json and raw_scores.npz.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--mlp-checkpoint", default=str(DEFAULT_MLP))
    parser.add_argument("--lstm-checkpoint", default=str(DEFAULT_LSTM))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    dataset = resolve(args.dataset)
    mlp_checkpoint = resolve(args.mlp_checkpoint)
    lstm_checkpoint = resolve(args.lstm_checkpoint)
    output = resolve(args.output)
    if not dataset.is_dir():
        raise SystemExit(f"dataset directory missing: {dataset}")
    for checkpoint in (mlp_checkpoint, lstm_checkpoint):
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint missing: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU fallback")
    device = torch.device("cuda")

    print(f"Loading exact SAFE split from {dataset}", flush=True)
    _cfg, all_rollouts, splits = load_exact_split(dataset)
    metadata = validate_rollouts(all_rollouts)
    train_rollouts = splits["train"]
    metadata.update({
        "dataset": str(dataset),
        "mlp_checkpoint": str(mlp_checkpoint),
        "lstm_checkpoint": str(lstm_checkpoint),
        "seed": 0,
        "unseen_task_ratio": 0.3,
        "seen_train_ratio": 0.6,
        "seen_task_ids": sorted({int(r.task_id) for r in splits["train"] + splits.get("val_seen", [])}),
        "unseen_task_ids": sorted({int(r.task_id) for r in splits.get("val_unseen", [])}),
        "train_rollouts": len(train_rollouts),
        "success_rollouts": sum(int(r.episode_success) for r in train_rollouts),
        "failure_rollouts": sum(1 - int(r.episode_success) for r in train_rollouts),
        "train_order_sha256": hashlib.sha256("\n".join(f"{r.task_id}:{r.episode_idx}:{int(r.episode_success)}" for r in train_rollouts).encode()).hexdigest(),
    })
    input_dim = int(train_rollouts[0].hidden_states.shape[-1])
    if input_dim != 4096:
        raise ValueError(f"unexpected training input dimension: {input_dim}")

    model_scores: dict[str, list[np.ndarray]] = {}
    reset_checks = {}
    for model_name, checkpoint in (("mlp", mlp_checkpoint), ("lstm", lstm_checkpoint)):
        print(f"Loading {model_name} checkpoint and scoring {len(train_rollouts)} train rollouts", flush=True)
        model_cfg = make_config(dataset, model_name)
        model = load_model(model_cfg, checkpoint, input_dim, device)
        scores, reset_verified = score_rollouts(model, train_rollouts, args.batch_size, device)
        model_scores[model_name] = scores
        reset_checks[model_name] = reset_verified
        print(f"  {model_name}: {sum(len(x) for x in scores)} native score steps; reset_verified={reset_verified}", flush=True)
        del model
        torch.cuda.empty_cache()

    records = []
    for index, rollout in enumerate(train_rollouts):
        length = int(rollout.hidden_states.shape[0])
        mlp_values = model_scores["mlp"][index]
        lstm_values = model_scores["lstm"][index]
        if len(mlp_values) != length or len(lstm_values) != length:
            raise ValueError("raw score length mismatch")
        records.append({
            "train_index": index,
            "task_id": int(rollout.task_id),
            "episode_idx": int(rollout.episode_idx),
            "episode_success": int(rollout.episode_success),
            "failure_label": 1 - int(rollout.episode_success),
            "task_description": str(rollout.task_description),
            "length": length,
            "mlp": {
                "scores": mlp_values.tolist(),
                "final": float(mlp_values[-1]),
                "max": float(mlp_values.max()),
                "mean": float(mlp_values.mean()),
                "final_per_step": float(mlp_values[-1] / length),
            },
            "lstm": {
                "scores": lstm_values.tolist(),
                "final": float(lstm_values[-1]),
                "max": float(lstm_values.max()),
                "mean": float(lstm_values.mean()),
            },
        })

    diagnostics = {"mlp": summarize(records, "mlp"), "lstm": summarize(records, "lstm")}
    representatives = {"mlp": {}, "lstm": {}}
    for model_name in representatives:
        for role, row in choose_representatives(records, model_name):
            representatives[model_name][role] = row

    lengths = np.asarray([r["length"] for r in records], dtype=np.int64)
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    np.savez_compressed(
        output / "raw_scores.npz",
        mlp_scores=np.concatenate([np.asarray(r["mlp"]["scores"], dtype=np.float32) for r in records]),
        lstm_scores=np.concatenate([np.asarray(r["lstm"]["scores"], dtype=np.float32) for r in records]),
        lengths=lengths,
        offsets=offsets,
        failure_labels=np.asarray([r["failure_label"] for r in records], dtype=np.int64),
    )
    write_json(output / "raw_scores.json", {
        "schema": "lf3r.safe.train_separability.v1",
        "metadata": metadata,
        "reset_verified": reset_checks,
        "rollouts": records,
    })
    write_json(output / "summary.json", {
        "schema": "lf3r.safe.train_separability.summary.v1",
        "metadata": metadata,
        "checkpoint_loading_strict": True,
        "reset_verified": reset_checks,
        "diagnostics": diagnostics,
        "representatives": representatives,
    })
    make_plots(output, records, representatives)
    make_report(output, metadata, diagnostics)
    print(json.dumps({
        "output": str(output),
        "train_rollouts": len(records),
        "success": metadata["success_rollouts"],
        "failure": metadata["failure_rollouts"],
        "reset_verified": reset_checks,
        "mlp_final_auroc": diagnostics["mlp"]["final"]["auroc_failure_positive"],
        "mlp_normalized_final_auroc": diagnostics["mlp"]["final_per_step"]["auroc_failure_positive"],
        "lstm_max_auroc": diagnostics["lstm"]["max"]["auroc_failure_positive"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
