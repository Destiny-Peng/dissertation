"""Train-only PCA and strict native-frame alignment for decoder latents."""
from __future__ import annotations
from pathlib import Path
import numpy as np

LATENT_MODES = ("robodopamine_latent", "robodopamine_latent_plus_fused")


def load_latent(path: Path, frames, sample_ids=None):
    with np.load(path, allow_pickle=False) as saved:
        features = np.asarray(saved["features"], dtype=np.float32)
        indices = saved["frame_indices"].tolist()
        ids = saved["sample_ids"].tolist()
        sample_indices = saved["sample_indices"].tolist()
        if str(saved["position"].item()) != "score_start":
            raise ValueError("Expected the final token of opening <score> (score_start); re-extract old latents")
        if str(saved["mode"].item()) != "incremental":
            raise ValueError("latent extraction must use incremental mode")
    if features.shape != (len(frames), 2560) or not np.isfinite(features).all():
        raise ValueError("latent features must be finite N x 2560")
    if indices != list(frames) or len(set(ids)) != len(ids):
        raise ValueError("latent native frame/sample alignment mismatch")
    if sample_indices != list(range(len(frames))):
        raise ValueError("latent sample indices are incomplete or out of order")
    if sample_ids is not None and ids != list(sample_ids):
        raise ValueError("latent sample IDs do not match incremental predictions")
    return features


def fit_pca(dataset, train_ids, components=64):
    x = np.concatenate([np.asarray(dataset[i]["sequence"], dtype=np.float32)[:, :2560]
                        for i in train_ids], axis=0)
    if x.shape[1] != 2560 or min(x.shape[0] - 1, x.shape[1]) < components:
        raise ValueError(f"PCA-{components} requires at least {components + 1} training samples")
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    return {"mean": mean, "components": vt[:components].copy(),
            "singular_values": singular[:components].copy(),
            "fit_rollout_ids": list(train_ids), "fit_sample_count": len(x)}


def transform_sequence(sequence, pca, plus_fused=False):
    x = np.asarray(sequence, dtype=np.float32)
    projected = (x[:, :2560] - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T
    return np.concatenate([projected, x[:, 2560:]], axis=1) if plus_fused else projected


def transform_dataset(dataset, pca, plus_fused=False):
    result = {}
    for key, row in dataset.items():
        result[key] = {**row, "sequence": transform_sequence(row["sequence"], pca, plus_fused)}
    return result
