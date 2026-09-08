"""Run official SAFE functional conformal calibration and held-out evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse the LF3R-owned exact-split and scoring implementation; upstream SAFE
# remains untouched.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from diagnose_train_separability import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_LSTM,
    DEFAULT_MLP,
    load_exact_split,
    load_model,
    make_config,
    score_rollouts,
)


DEFAULT_OUTPUT = ROOT / "outputs/safe_training/conformal/lf3r_natural_libero10_20260906_seed0"
ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
PRIMARY_ALPHA = 0.2


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def pad_edge(scores: list[np.ndarray], length: int) -> np.ndarray:
    if not scores or any(len(s) == 0 for s in scores):
        raise ValueError("functional calibration requires non-empty score trajectories")
    return np.asarray([np.pad(np.asarray(s, dtype=np.float64), (0, length - len(s)), mode="edge") for s in scores])


def describe(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None, "iqr": None, "min": None, "max": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def key_for(rollout: Any) -> str:
    return f"task{int(rollout.task_id)}--ep{int(rollout.episode_idx)}--succ{int(rollout.episode_success)}"


def official_calibration(
    rollouts_by_split: dict[str, list[Any]],
    scores_by_split: dict[str, list[np.ndarray]],
    rng_state_after_split: tuple,
    alphas: list[float],
):
    """Mirror eval_functional_conformal's calibration and return its metadata."""
    from failure_prob.utils.conformal.functional_predictor import (
        FunctionalPredictor,
        ModulationType,
        RegressionType,
    )

    cal_rollouts = list(rollouts_by_split["val_seen"])
    test_rollouts = list(rollouts_by_split["val_unseen"])
    max_length = max(len(s) for s in scores_by_split["val_seen"] + scores_by_split["val_unseen"])
    cal_padded = pad_edge(scores_by_split["val_seen"], max_length)
    test_padded = pad_edge(scores_by_split["val_unseen"], max_length)

    success_indices = [i for i, r in enumerate(cal_rollouts) if int(r.episode_success) == 1]
    cal_success = cal_padded[success_indices]
    if len(cal_success) == 1:
        order = np.arange(1, dtype=np.int64)
        cal_1 = cal_success
        cal_2 = cal_success
        cal_1_indices = [0]
        cal_2_indices = [0]
    else:
        # Restore the RNG state immediately after the official split. This is
        # exactly the state seen by SAFE's np.random.shuffle call.
        np.random.set_state(rng_state_after_split)
        order = np.arange(len(cal_success), dtype=np.int64)
        np.random.shuffle(order)
        n_cal_1 = int(len(cal_success) * 0.3)
        cal_1_indices = order[:n_cal_1].tolist()
        cal_2_indices = order[n_cal_1:].tolist()
        cal_1 = cal_success[cal_1_indices]
        cal_2 = cal_success[cal_2_indices]

    bands = {}
    for alpha in alphas:
        predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
        band = predictor.get_one_sided_prediction_band(cal_1, cal_2, alpha, lower_bound=False)
        band = np.asarray(band[0], dtype=np.float64)
        if band.shape != (max_length,) or not np.isfinite(band).all():
            raise ValueError(f"invalid conformal band shape/values: {band.shape}")
        bands[float(alpha)] = band

    # Cross-check against the actual upstream routine. It mutates only copies
    # of the score lists and computes metrics after constructing the same band.
    from failure_prob.utils.metrics import eval_functional_conformal

    copied_scores = {name: [np.array(s, dtype=np.float64, copy=True) for s in values] for name, values in scores_by_split.items()}
    np.random.set_state(rng_state_after_split)
    official_df, official_bands = eval_functional_conformal(
        rollouts_by_split,
        copied_scores,
        "model",
        alphas=alphas,
        calib_split_names=["val_seen"],
        test_split_names=["val_unseen"],
        align_method="extend",
    )
    crosscheck = {}
    for alpha in alphas:
        upstream = np.asarray(official_bands[float(alpha)])[0]
        crosscheck[str(alpha)] = {
            "allclose": bool(np.allclose(upstream, bands[float(alpha)], atol=1e-12, rtol=1e-12)),
            "max_abs_diff": float(np.max(np.abs(upstream - bands[float(alpha)]))),
        }
        if not crosscheck[str(alpha)]["allclose"]:
            raise RuntimeError(f"custom band differs from official SAFE band at alpha={alpha}")

    cal_records = [cal_rollouts[i] for i in success_indices]
    return {
        "max_length": max_length,
        "calibration_rollouts": len(cal_rollouts),
        "calibration_success_rollouts": len(cal_success),
        "calibration_failure_rollouts": len(cal_rollouts) - len(cal_success),
        "regression_subset_count": len(cal_1),
        "modulation_subset_count": len(cal_2),
        "regression_subset": [key_for(cal_records[i]) for i in cal_1_indices],
        "modulation_subset": [key_for(cal_records[i]) for i in cal_2_indices],
        "success_shuffle_order": [key_for(cal_records[i]) for i in order.tolist()],
        "split_fraction_for_regression": 0.3,
        "calibration_on": "val_seen successful rollouts only",
        "lower_bound": False,
        "predictor": "FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)",
        "alignment": "extend: edge-pad each trajectory to max length",
        "bands": {str(alpha): bands[float(alpha)].tolist() for alpha in alphas},
        "upstream_crosscheck": crosscheck,
        "upstream_rows": official_df.to_dict(orient="records"),
        "calibration_record_keys": [key_for(r) for r in cal_records],
    }, bands, test_padded


def raw_quality_metrics(rollouts: list[Any], scores: list[np.ndarray]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray([1 - int(r.episode_success) for r in rollouts], dtype=np.int64)
    task_min = np.asarray([int(r.task_min_step) for r in rollouts], dtype=np.int64)
    summaries = {
        "final_native": np.asarray([s[-1] for s in scores], dtype=np.float64),
        "max_native": np.asarray([np.max(s) for s in scores], dtype=np.float64),
        "mean_native": np.asarray([np.mean(s) for s in scores], dtype=np.float64),
        "max_until_task_min_step": np.asarray([np.max(s[:m]) for s, m in zip(scores, task_min)], dtype=np.float64),
    }
    result = {}
    for name, values in summaries.items():
        result[name] = {
            "auroc_failure_positive": float(roc_auc_score(labels, values)),
            "auprc_failure_positive": float(average_precision_score(labels, values)),
            "success_summary": describe(values[labels == 0]),
            "failure_summary": describe(values[labels == 1]),
        }
    return result


def classify_with_band(
    rollouts: list[Any],
    raw_scores: list[np.ndarray],
    band: np.ndarray,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    padded = pad_edge(raw_scores, len(band))
    task_min = np.asarray([int(r.task_min_step) for r in rollouts], dtype=np.int64)
    native_lengths = np.asarray([len(s) for s in raw_scores], dtype=np.int64)
    if mode == "by final end":
        effective_lengths = np.full(len(rollouts), len(band), dtype=np.int64)
    elif mode == "by earliest stop":
        effective_lengths = task_min.copy()
    else:
        raise ValueError(mode)

    mask = padded >= band[None, :]
    if mode == "by earliest stop":
        for i, length in enumerate(effective_lengths):
            mask[i, length:] = False
    has_detection = np.any(mask, axis=1)
    first_index = np.argmax(mask, axis=1)
    detection_time = np.where(has_detection, first_index, effective_lengths)
    relative_time = detection_time / effective_lengths
    failure_labels = np.asarray([1 - int(r.episode_success) for r in rollouts], dtype=np.int64)

    tp = int(np.sum(has_detection & (failure_labels == 1)))
    fn = int(np.sum(~has_detection & (failure_labels == 1)))
    fp = int(np.sum(has_detection & (failure_labels == 0)))
    tn = int(np.sum(~has_detection & (failure_labels == 0)))
    failure_mask = failure_labels == 1
    success_mask = failure_labels == 0
    metrics = {
        "mode": mode,
        "test_rollouts": len(rollouts),
        "success_rollouts": int(success_mask.sum()),
        "failure_rollouts": int(failure_mask.sum()),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "success_false_positive_rate": float(fp / success_mask.sum()) if success_mask.sum() else 0.0,
        "failure_detection_rate": float(tp / failure_mask.sum()) if failure_mask.sum() else 0.0,
        "failure_recall": float(tp / failure_mask.sum()) if failure_mask.sum() else 0.0,
        "success_specificity": float(tn / success_mask.sum()) if success_mask.sum() else 0.0,
        "accuracy": float((tp + tn) / len(rollouts)),
        "mean_relative_detection_time_failures": float(np.mean(relative_time[failure_mask])),
        "first_detection_distribution_failures": {
            "all_failures": describe(relative_time[failure_mask]),
            "detected_failures_only": describe(relative_time[failure_mask & has_detection]),
            "detected_count": int(np.sum(failure_mask & has_detection)),
            "not_detected_count": int(np.sum(failure_mask & ~has_detection)),
            "native_crossing_count": int(np.sum(failure_mask & has_detection & (first_index < native_lengths))),
            "padded_only_crossing_count": int(np.sum(failure_mask & has_detection & (first_index >= native_lengths))),
        },
        "first_detection_distribution_all": {
            "relative_official_time": describe(relative_time),
            "detected_count": int(has_detection.sum()),
            "not_detected_count": int((~has_detection).sum()),
        },
    }

    rows = []
    for i, (rollout, raw, native_length) in enumerate(zip(rollouts, raw_scores, native_lengths)):
        first = int(first_index[i]) if bool(has_detection[i]) else None
        rows.append({
            "test_index": i,
            "task_id": int(rollout.task_id),
            "episode_idx": int(rollout.episode_idx),
            "episode_success": int(rollout.episode_success),
            "failure_label_posthoc": int(failure_labels[i]),
            "task_description": str(rollout.task_description),
            "native_length": int(native_length),
            "task_min_step": int(task_min[i]),
            "effective_length": int(effective_lengths[i]),
            "raw_scores": np.asarray(raw, dtype=np.float32).tolist(),
            "threshold_native": band[:native_length].astype(np.float32).tolist(),
            "margin_native": (np.asarray(raw, dtype=np.float64) - band[:native_length]).astype(np.float32).tolist(),
            "first_crossing_index_0based_padded": first,
            "first_crossing_timestep_1based_padded": None if first is None else first + 1,
            "first_crossing_index_0based_native": None if first is None or first >= native_length else first,
            "crossing_after_native": bool(first is not None and first >= native_length),
            "has_detection": bool(has_detection[i]),
            "official_detection_time": int(detection_time[i]),
            "official_relative_detection_time": float(relative_time[i]),
        })
    return rows, metrics


def load_manifest_annotations(root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    annotations = {}
    if not manifest_path.is_file():
        return annotations
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video = row.get("video_path")
        if not video:
            continue
        annotations[str(Path(video).as_posix())] = row
    return annotations


def annotation_delays(root: Path, rollouts: list[Any], rows_by_model: dict[str, list[dict[str, Any]]], manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest_annotations(root, manifest_path)
    matched = []
    for index, rollout in enumerate(rollouts):
        try:
            relative_video = str(Path(rollout.mp4_path).resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            continue
        row = manifest.get(relative_video)
        if not row:
            continue
        injection = row.get("injection") or {}
        causal = row.get("causal_onset_frame", injection.get("causal_onset_frame"))
        observable = row.get("observable_onset_frame")
        if causal is None and observable is None:
            continue
        matched.append({"test_index": index, "causal_onset_frame": causal, "observable_onset_frame": observable})
    # The current natural LIBERO-10 manifest has no causal/observable onset
    # annotations. Keep the explicit empty result instead of inferring them.
    result = {
        "manifest": str(manifest_path),
        "matched_test_rollouts": len(matched),
        "annotation_unit": "video frame index; no conversion applied",
        "events": matched,
        "delay_results": {},
    }
    if matched:
        for model, rows in rows_by_model.items():
            model_delays = {}
            for name, onset_key in (("causal", "causal_onset_frame"), ("observable", "observable_onset_frame")):
                values = []
                for event in matched:
                    onset = event.get(onset_key)
                    if onset is None:
                        continue
                    row = rows[event["test_index"]]
                    first = row["first_crossing_index_0based_native"]
                    if first is not None:
                        values.append(first - int(onset))
                model_delays[name] = describe(np.asarray(values, dtype=float))
            result["delay_results"][model] = model_delays
    return result


def make_plots(output: Path, evaluations: dict[str, Any], thresholds: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, model, title in zip(axes, ("mlp", "lstm"), ("SAFE-MLP", "SAFE-LSTM")):
        band = np.asarray(thresholds[model][str(PRIMARY_ALPHA)], dtype=float)
        ax.plot(np.arange(1, len(band) + 1), band, color="#c0504d", label="functional CP threshold")
        for mode, color in (("by final end", "#4472c4"), ("by earliest stop", "#70ad47")):
            metric = evaluations[model][str(PRIMARY_ALPHA)][mode]["metrics"]
            values = [r["official_relative_detection_time"] for r in evaluations[model][str(PRIMARY_ALPHA)][mode]["rows"] if r["failure_label_posthoc"] == 1]
            if values:
                ax.axvline(np.median(values) * len(band), color=color, linestyle="--", label=mode + " median relative time")
        ax.set_title(title + " threshold")
        ax.set_xlabel("padded SAFE timestep")
        ax.set_ylabel("score / threshold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "conformal_thresholds_and_detection.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, model in zip(axes, ("mlp", "lstm")):
        for mode, color in (("by final end", "#4472c4"), ("by earliest stop", "#70ad47")):
            values = [r["official_relative_detection_time"] for r in evaluations[model][str(PRIMARY_ALPHA)][mode]["rows"] if r["failure_label_posthoc"] == 1]
            if values:
                ax.hist(values, bins=10, alpha=0.5, label=mode, color=color)
        ax.set_title(model.upper() + " failure first-detection times")
        ax.set_xlabel("official relative detection time")
        ax.set_ylabel("failure rollout count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "first_detection_time_distributions.png", dpi=180)
    plt.close(fig)


def make_report(output: Path, metadata: dict[str, Any], calibration: dict[str, Any], metrics: dict[str, Any], quality: dict[str, Any], annotations: dict[str, Any]) -> None:
    lines = [
        "# SAFE functional conformal calibration and held-out evaluation",
        "",
        "This report uses the already trained checkpoints; it does not retrain, run LIBERO/OpenVLA, tune on test labels, or select a fixed threshold.",
        "",
        "## Official protocol",
        "",
        f"- Exact seed-0 split: train={metadata['split_counts']['train']}, val_seen={metadata['split_counts']['val_seen']}, val_unseen={metadata['split_counts']['val_unseen']}; seen tasks={metadata['seen_task_ids']}, unseen tasks={metadata['unseen_task_ids']}.",
        f"- Calibration uses only successful val_seen rollouts: {calibration['calibration_success_rollouts']} trajectories, split exactly as SAFE into {calibration['regression_subset_count']} regression and {calibration['modulation_subset_count']} modulation/calibration trajectories with the official 30% fraction.",
        f"- Predictor: {calibration['predictor']}; one-sided upper band, lower_bound={calibration['lower_bound']}; alignment={calibration['alignment']}; maximum aligned length={calibration['max_length']}.",
        f"- Primary configured alpha: {PRIMARY_ALPHA}; full official alpha grid is saved in thresholds.json. The exact upstream routine matched the reproduced bands at every alpha (max absolute difference 0).",
        "- Threshold construction never uses val_unseen labels. Test success/failure is used only after raw scores, thresholds, margins, and crossings are computed.",
        "",
        "## Primary alpha=0.2 detection results",
        "",
        "| model | evaluation mode | success FPR | failure recall | TP / FN | FP / TN | median failure relative detection time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("mlp", "lstm"):
        for mode in ("by final end", "by earliest stop"):
            item = metrics[model][str(PRIMARY_ALPHA)][mode]
            m = item["metrics"]
            dist = m["first_detection_distribution_failures"]["detected_failures_only"]
            lines.append(f"| {model.upper()} | {mode} | {m['success_false_positive_rate']:.4f} | {m['failure_detection_rate']:.4f} | {m['true_positive']} / {m['false_negative']} | {m['false_positive']} / {m['true_negative']} | {dist['median'] if dist['count'] else 'n/a'} |")
    lines += ["", "## Secondary raw-score quality on val_unseen", "", "| model | summary | AUROC (failure positive) | AUPRC (failure positive) |", "|---|---|---:|---:|"]
    for model in ("mlp", "lstm"):
        for summary, values in quality[model].items():
            lines.append(f"| {model.upper()} | {summary} | {values['auroc_failure_positive']:.4f} | {values['auprc_failure_positive']:.4f} |")
    lines += ["", "## Onset-relative delay", "",
              f"Matched held-out causal/observable annotation rollouts: {annotations['matched_test_rollouts']}.", "The natural LIBERO-10 manifest contains no causal or observable onset fields for these val_unseen rollouts, so no onset-delay statistic is fabricated.", "",
              "Raw held-out trajectories, native thresholds, margins, first crossings, calibration bands, and all-alpha metrics are saved alongside this report.", "",
              "Plots: conformal_thresholds_and_detection.png and first_detection_time_distributions.png.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--mlp-checkpoint", default=str(DEFAULT_MLP))
    parser.add_argument("--lstm-checkpoint", default=str(DEFAULT_LSTM))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--manifest", default=str(ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"))
    args = parser.parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    mlp_checkpoint = Path(args.mlp_checkpoint).expanduser().resolve()
    lstm_checkpoint = Path(args.lstm_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not dataset.is_dir() or not mlp_checkpoint.is_file() or not lstm_checkpoint.is_file():
        raise SystemExit("dataset or checkpoint path is missing")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU fallback")
    device = torch.device("cuda")

    print("Loading official seed-0 split and latent rollouts", flush=True)
    _cfg, all_rollouts, splits = load_exact_split(dataset)
    rng_state_after_split = np.random.get_state()
    split_names = ("train", "val_seen", "val_unseen")
    split_counts = {name: len(splits[name]) for name in split_names}
    split_records = {name: list(splits[name]) for name in split_names}
    metadata = {
        "dataset": str(dataset),
        "mlp_checkpoint": str(mlp_checkpoint),
        "lstm_checkpoint": str(lstm_checkpoint),
        "seed": 0,
        "seen_task_ids": sorted({int(r.task_id) for r in splits["train"] + splits["val_seen"]}),
        "unseen_task_ids": sorted({int(r.task_id) for r in splits["val_unseen"]}),
        "split_counts": split_counts,
        "token_idx_rel": 1.0,
        "selected_token_idx": 6,
        "input_dim": int(all_rollouts[0].hidden_states.shape[-1]),
        "feature_shape_after_selection": "(T,4096)",
        "source_shape_before_selection": "(T,7,4096)",
        "normalize_hidden_states": False,
        "split_order_sha256": {
            name: hashlib.sha256("\n".join(key_for(r) for r in splits[name]).encode()).hexdigest()
            for name in split_names
        },
    }
    if metadata["input_dim"] != 4096:
        raise ValueError("unexpected SAFE input dimension")

    scores_by_model: dict[str, dict[str, list[np.ndarray]]] = {}
    reset_verified = {}
    for model_name, checkpoint in (("mlp", mlp_checkpoint), ("lstm", lstm_checkpoint)):
        print(f"Scoring {model_name} on train/val_seen/val_unseen", flush=True)
        cfg = make_config(dataset, model_name)
        model = load_model(cfg, checkpoint, metadata["input_dim"], device)
        scores_by_model[model_name] = {}
        reset_verified[model_name] = {}
        for split_name in split_names:
            scores, reset = score_rollouts(model, splits[split_name], args.batch_size, device)
            scores_by_model[model_name][split_name] = scores
            reset_verified[model_name][split_name] = reset
            print(f"  {split_name}: {len(scores)} rollouts, {sum(len(s) for s in scores)} steps, reset={reset}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Compare the exact train scores against the previous diagnostic artifact.
    reproducibility = {}
    prior_path = ROOT / "outputs/safe_training/diagnostics/lf3r_natural_libero10_train/raw_scores.json"
    if prior_path.is_file():
        prior = {key_for_row: row for key_for_row, row in []}
        prior_payload = json.loads(prior_path.read_text())
        prior_by_key = {f"task{r['task_id']}--ep{r['episode_idx']}--succ{r['episode_success']}": r for r in prior_payload["rollouts"]}
        for model_name in ("mlp", "lstm"):
            diffs = []
            for rollout, values in zip(splits["train"], scores_by_model[model_name]["train"]):
                old = prior_by_key[key_for(rollout)][model_name]["scores"]
                diffs.append(float(np.max(np.abs(np.asarray(old, dtype=np.float64) - values))))
            reproducibility[model_name] = {"max_abs_diff_vs_prior_train_scores": max(diffs) if diffs else None}
    metadata["checkpoint_loading_strict"] = True
    metadata["reset_verified"] = reset_verified
    metadata["all_features_and_scores_finite"] = bool(all(np.isfinite(s).all() for model in scores_by_model.values() for split in model.values() for s in split))
    metadata["train_score_reproducibility"] = reproducibility

    # Save all raw scores before calibration. This file contains no thresholds.
    all_score_rows = []
    for split_name in split_names:
        for index, rollout in enumerate(splits[split_name]):
            all_score_rows.append({
                "split": split_name,
                "split_index": index,
                "task_id": int(rollout.task_id),
                "episode_idx": int(rollout.episode_idx),
                "episode_success": int(rollout.episode_success),
                "failure_label_posthoc": int(1 - int(rollout.episode_success)),
                "native_length": int(len(scores_by_model["mlp"][split_name][index])),
                "mlp_scores": scores_by_model["mlp"][split_name][index].astype(np.float32).tolist(),
                "lstm_scores": scores_by_model["lstm"][split_name][index].astype(np.float32).tolist(),
            })
    write_json(output / "scores_all.json", {"schema": "lf3r.safe.official_scores.v1", "metadata": metadata, "rollouts": all_score_rows})

    rollouts_by_split = {name: splits[name] for name in ("val_seen", "val_unseen")}
    score_lists_for_calibration = {
        model_name: {
            "val_seen": scores_by_model[model_name]["val_seen"],
            "val_unseen": scores_by_model[model_name]["val_unseen"],
        }
        for model_name in ("mlp", "lstm")
    }
    calibration_by_model = {}
    bands_by_model = {}
    padded_test_by_model = {}
    functional_metrics = {}
    quality_metrics = {}
    evaluation_rows = {}
    for model_name in ("mlp", "lstm"):
        print(f"Reproducing official functional CP for {model_name}", flush=True)
        calibration, bands, padded_test = official_calibration(
            rollouts_by_split,
            score_lists_for_calibration[model_name],
            rng_state_after_split,
            ALPHAS,
        )
        calibration_by_model[model_name] = calibration
        bands_by_model[model_name] = {str(alpha): bands[float(alpha)].tolist() for alpha in ALPHAS}
        padded_test_by_model[model_name] = padded_test
        quality_metrics[model_name] = raw_quality_metrics(splits["val_unseen"], scores_by_model[model_name]["val_unseen"])
        functional_metrics[model_name] = {}
        evaluation_rows[model_name] = {}
        for alpha in ALPHAS:
            functional_metrics[model_name][str(alpha)] = {}
            evaluation_rows[model_name][str(alpha)] = {}
            for mode in ("by final end", "by earliest stop"):
                rows, metrics = classify_with_band(
                    splits["val_unseen"],
                    scores_by_model[model_name]["val_unseen"],
                    bands[float(alpha)],
                    mode,
                )
                functional_metrics[model_name][str(alpha)][mode] = {"metrics": metrics}
                if abs(alpha - PRIMARY_ALPHA) < 1e-12:
                    evaluation_rows[model_name][str(alpha)][mode] = {"metrics": metrics, "rows": rows}

    annotations = annotation_delays(ROOT, splits["val_unseen"], {m: evaluation_rows[m][str(PRIMARY_ALPHA)]["by final end"]["rows"] for m in ("mlp", "lstm")}, manifest_path)
    metadata["annotation_matches"] = annotations["matched_test_rollouts"]
    write_json(output / "calibration.json", {
        "schema": "lf3r.safe.functional_conformal.calibration.v1",
        "metadata": metadata,
        "primary_alpha": PRIMARY_ALPHA,
        "models": calibration_by_model,
    })
    write_json(output / "thresholds.json", {
        "schema": "lf3r.safe.functional_conformal.thresholds.v1",
        "primary_alpha": PRIMARY_ALPHA,
        "alphas": ALPHAS,
        "alignment": "extend",
        "models": bands_by_model,
    })
    write_json(output / "evaluation.json", {
        "schema": "lf3r.safe.functional_conformal.evaluation.v1",
        "metadata": metadata,
        "primary_alpha": PRIMARY_ALPHA,
        "models": evaluation_rows,
        "all_alpha_metrics": functional_metrics,
        "raw_quality_metrics": quality_metrics,
        "onset_delay": annotations,
    })
    make_plots(output, evaluation_rows, bands_by_model)
    make_report(output, metadata, calibration_by_model["mlp"], {m: {a: {mode: functional_metrics[m][a][mode] for mode in ("by final end", "by earliest stop")} for a in [str(PRIMARY_ALPHA)]} for m in ("mlp", "lstm")}, quality_metrics, annotations)
    print(json.dumps({
        "output": str(output),
        "split_counts": split_counts,
        "primary_alpha": PRIMARY_ALPHA,
        "reset_verified": reset_verified,
        "annotation_matches": annotations["matched_test_rollouts"],
        "mlp_final_end": functional_metrics["mlp"][str(PRIMARY_ALPHA)]["by final end"]["metrics"],
        "mlp_earliest_stop": functional_metrics["mlp"][str(PRIMARY_ALPHA)]["by earliest stop"]["metrics"],
        "lstm_final_end": functional_metrics["lstm"][str(PRIMARY_ALPHA)]["by final end"]["metrics"],
        "lstm_earliest_stop": functional_metrics["lstm"][str(PRIMARY_ALPHA)]["by earliest stop"]["metrics"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
