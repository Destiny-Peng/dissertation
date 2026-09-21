#!/usr/bin/env python3
"""Train lightweight failure-localization heads on saved Robo-Dopamine fused signals.

CPU-only probe. Reuses saved fused progress/hop and current failure annotations;
never launches Robo-Dopamine. Splits are strictly by rollout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from robo_incremental_hop.io import (
    PROJECT_ROOT,
    build_base_records,
    ensure_within_project,
    load_manifest,
    project_relative,
    resolve_project_path,
)
from robo_incremental_hop.localization_ranking import offline_change_point_localization
from robo_incremental_hop.phenotypes import build_phenotype_detector_configs
from robo_incremental_hop.report import (
    build_pairwise_ensemble_rows,
    evaluate_all_configs,
    write_csv,
)
from robo_incremental_hop.search_cache import load_search_cache, search_fingerprint

DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "annotations/failure_annotations/v1/records"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/robo_dopamine_localization_head"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "outputs/robo_dopamine_incremental_hop"
MODEL_NAMES = (
    "linear_probe",
    "tiny_mlp",
    "tiny_cnn",
    "tiny_bilstm_h16",
    "tiny_bilstm_h32",
)
SEQUENCE_MODEL_NAMES = {"tiny_bilstm_h16", "tiny_bilstm_h32"}
METRIC_NAMES = (
    "in_interval_rate",
    "within_1",
    "within_3",
    "within_5",
    "before_interval_rate",
    "after_interval_rate",
    "median_absolute_interval_error_samples",
    "mae_samples",
    "mse_samples",
)


def _first_primary_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        if str(raw.get("outcome") or "") != "terminal_failure":
            continue
        causal = raw.get("causal_onset_frame")
        observable = raw.get("observable_onset_frame")
        if causal is None or observable is None:
            continue
        causal_i, observable_i = int(causal), int(observable)
        if causal_i > observable_i:
            continue
        row = dict(raw)
        row["causal_onset_frame"] = causal_i
        row["observable_onset_frame"] = observable_i
        row["event_index"] = int(row.get("event_index") or 0)
        row["event_id"] = str(
            row.get("event_id")
            or f"{row['rollout_id']}::event{row['event_index']}"
        )
        by_rollout[str(row["rollout_id"])].append(row)
    primary = [
        min(
            rows,
            key=lambda row: (
                int(row["causal_onset_frame"]),
                int(row["observable_onset_frame"]),
                int(row["event_index"]),
            ),
        )
        for rows in by_rollout.values()
    ]
    return sorted(primary, key=lambda row: str(row["rollout_id"]))


def _onset_anchor(frames: Sequence[int], frame: int) -> int | None:
    return next(
        (index for index, value in enumerate(frames) if int(value) >= int(frame)),
        None,
    )


def _context(signal: Mapping[str, Any], radius: int) -> np.ndarray:
    progress = np.asarray(signal["progress"], dtype=np.float64)
    hops = np.asarray(signal["hops"], dtype=np.float64)
    if progress.ndim != 1 or hops.ndim != 1 or len(progress) != len(hops):
        raise ValueError("progress/hops must be same-length 1D arrays")
    if len(progress) == 0:
        raise ValueError("empty fused signal")
    stacked = np.stack([progress, hops], axis=0)
    padded = np.pad(stacked, ((0, 0), (radius, radius)), mode="edge")
    width = 2 * radius + 1
    return np.stack(
        [padded[:, index : index + width] for index in range(len(progress))],
        axis=0,
    )


def build_dataset(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    radius: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in _first_primary_events(events):
        rollout_id = str(event["rollout_id"])
        signal = signals.get(rollout_id)
        if signal is None:
            continue
        frames = [int(value) for value in signal["frames"]]
        causal = _onset_anchor(frames, int(event["causal_onset_frame"]))
        observable = _onset_anchor(frames, int(event["observable_onset_frame"]))
        if causal is None or observable is None or causal > observable:
            continue
        labels = np.zeros(len(frames), dtype=np.float64)
        labels[causal : observable + 1] = 1.0
        result[rollout_id] = {
            "event": event,
            "frames": np.asarray(frames, dtype=np.int64),
            "context": _context(signal, radius),
            "labels": labels,
            "causal_index": causal,
            "observable_index": observable,
            "progress": np.asarray(signal["progress"], dtype=np.float64),
            "sequence": np.stack(
                [
                    np.asarray(signal["progress"], dtype=np.float64),
                    np.asarray(signal["hops"], dtype=np.float64),
                ],
                axis=1,
            ),
        }
    return result


def flatten_dataset(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    xs = [
        np.asarray(dataset[rollout_id]["context"], dtype=np.float64)
        for rollout_id in rollout_ids
    ]
    ys = [
        np.asarray(dataset[rollout_id]["labels"], dtype=np.float64)
        for rollout_id in rollout_ids
    ]
    if not xs:
        raise ValueError("empty rollout selection")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def standardization_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def weighted_bce(logits: np.ndarray, y: np.ndarray, pos_weight: float) -> float:
    weights = np.where(y > 0.5, pos_weight, 1.0)
    loss = (
        np.maximum(logits, 0.0)
        - logits * y
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(np.sum(weights * loss) / max(1e-12, np.sum(weights)))


def _pos_weight(y: np.ndarray) -> float:
    positive = float(np.sum(y > 0.5))
    negative = float(np.sum(y <= 0.5))
    if positive <= 0:
        raise ValueError("training split contains no positive interval samples")
    return min(20.0, max(1.0, negative / positive))


class Adam:
    def __init__(self, params: Mapping[str, np.ndarray], lr: float) -> None:
        self.lr = lr
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}
        self.t = 0

    def step(
        self,
        params: Mapping[str, np.ndarray],
        grads: Mapping[str, np.ndarray],
    ) -> None:
        self.t += 1
        for key in params:
            grad = grads[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * (grad * grad)
            m_hat = self.m[key] / (1.0 - 0.9 ** self.t)
            v_hat = self.v[key] / (1.0 - 0.999 ** self.t)
            params[key][...] -= (
                self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            )


class BaseHead:
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def logits(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearProbe(BaseHead):
    def __init__(self, *, seed: int, epochs: int, patience: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.epochs = epochs
        self.patience = patience
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def _flat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(len(x), -1)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        xf, xvf = self._flat(x), self._flat(x_val)
        self.w = self.rng.normal(0.0, 0.01, size=(xf.shape[1],))
        self.b = np.zeros(1, dtype=np.float64)
        params = {"w": self.w, "b": self.b}
        optimizer = Adam(params, lr=0.03)
        pos_weight = _pos_weight(y)
        best_loss = math.inf
        best_w = None
        best_b = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, self.epochs + 1):
            logits = xf @ self.w + self.b[0]
            probability = _sigmoid(logits)
            sample_weight = np.where(y > 0.5, pos_weight, 1.0)
            dz = (
                sample_weight * (probability - y)
                / max(1e-12, float(np.sum(sample_weight)))
            )
            grads = {
                "w": xf.T @ dz + 1e-4 * self.w,
                "b": np.asarray([np.sum(dz)], dtype=np.float64),
            }
            optimizer.step(params, grads)
            val_loss = weighted_bce(
                xvf @ self.w + self.b[0],
                y_val,
                pos_weight,
            )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_w = self.w.copy()
                best_b = self.b.copy()
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_w is not None and best_b is not None:
            self.w[...] = best_w
            self.b[...] = best_b
        return {
            "best_val_bce": best_loss,
            "best_epoch": best_epoch,
            "pos_weight": pos_weight,
        }

    def logits(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None and self.b is not None
        return self._flat(x) @ self.w + self.b[0]


class TinyMLP(BaseHead):
    def __init__(
        self,
        *,
        seed: int,
        epochs: int,
        patience: int,
        hidden: int = 16,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.epochs = epochs
        self.patience = patience
        self.hidden = hidden
        self.params: dict[str, np.ndarray] = {}

    def _flat(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(len(x), -1)

    def _forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(z1, 0.0)
        logits = hidden @ self.params["w2"] + self.params["b2"][0]
        return z1, hidden, logits

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        xf, xvf = self._flat(x), self._flat(x_val)
        scale = math.sqrt(2.0 / max(1, xf.shape[1]))
        self.params = {
            "w1": self.rng.normal(
                0.0,
                scale,
                size=(xf.shape[1], self.hidden),
            ),
            "b1": np.zeros(self.hidden, dtype=np.float64),
            "w2": self.rng.normal(
                0.0,
                math.sqrt(2.0 / self.hidden),
                size=(self.hidden,),
            ),
            "b2": np.zeros(1, dtype=np.float64),
        }
        optimizer = Adam(self.params, lr=0.01)
        pos_weight = _pos_weight(y)
        best_loss = math.inf
        best_params = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, self.epochs + 1):
            z1, hidden, logits = self._forward(xf)
            probability = _sigmoid(logits)
            sample_weight = np.where(y > 0.5, pos_weight, 1.0)
            dz2 = (
                sample_weight * (probability - y)
                / max(1e-12, float(np.sum(sample_weight)))
            )
            dh = dz2[:, None] * self.params["w2"][None, :]
            dz1 = dh * (z1 > 0.0)
            grads = {
                "w2": hidden.T @ dz2 + 1e-4 * self.params["w2"],
                "b2": np.asarray([np.sum(dz2)], dtype=np.float64),
                "w1": xf.T @ dz1 + 1e-4 * self.params["w1"],
                "b1": np.sum(dz1, axis=0),
            }
            optimizer.step(self.params, grads)
            val_loss = weighted_bce(
                self._forward(xvf)[2],
                y_val,
                pos_weight,
            )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_params = {
                    key: value.copy()
                    for key, value in self.params.items()
                }
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_params is not None:
            self.params = best_params
        return {
            "best_val_bce": best_loss,
            "best_epoch": best_epoch,
            "pos_weight": pos_weight,
        }

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(self._flat(x))[2]


class TinyCNN(BaseHead):
    def __init__(
        self,
        *,
        seed: int,
        epochs: int,
        patience: int,
        filters: int = 8,
        kernel: int = 3,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.epochs = epochs
        self.patience = patience
        self.filters = filters
        self.kernel = kernel
        self.params: dict[str, np.ndarray] = {}

    def _windows(self, x: np.ndarray) -> np.ndarray:
        _, _, width = x.shape
        if width < self.kernel:
            raise ValueError("context width is smaller than CNN kernel")
        return np.stack(
            [
                x[:, :, index : index + self.kernel]
                for index in range(width - self.kernel + 1)
            ],
            axis=1,
        )

    def _forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        windows = self._windows(x)
        conv = (
            np.einsum("npck,fck->npf", windows, self.params["wc"])
            + self.params["bc"][None, None, :]
        )
        activation = np.maximum(conv, 0.0)
        logits = (
            np.einsum("npf,pf->n", activation, self.params["wo"])
            + self.params["bo"][0]
        )
        return windows, conv, activation, logits

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        channels = x.shape[1]
        self.params = {
            "wc": self.rng.normal(
                0.0,
                math.sqrt(2.0 / (channels * self.kernel)),
                size=(self.filters, channels, self.kernel),
            ),
            "bc": np.zeros(self.filters, dtype=np.float64),
            "wo": self.rng.normal(
                0.0,
                math.sqrt(2.0 / self.filters),
                size=(x.shape[2] - self.kernel + 1, self.filters),
            ),
            "bo": np.zeros(1, dtype=np.float64),
        }
        optimizer = Adam(self.params, lr=0.01)
        pos_weight = _pos_weight(y)
        best_loss = math.inf
        best_params = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, self.epochs + 1):
            windows, conv, activation, logits = self._forward(x)
            probability = _sigmoid(logits)
            sample_weight = np.where(y > 0.5, pos_weight, 1.0)
            dz = (
                sample_weight * (probability - y)
                / max(1e-12, float(np.sum(sample_weight)))
            )
            dactivation = dz[:, None, None] * self.params["wo"][None, :, :]
            dconv = dactivation * (conv > 0.0)
            grads = {
                "wo": (
                    np.einsum("n,npf->pf", dz, activation)
                    + 1e-4 * self.params["wo"]
                ),
                "bo": np.asarray([np.sum(dz)], dtype=np.float64),
                "wc": (
                    np.einsum("npf,npck->fck", dconv, windows)
                    + 1e-4 * self.params["wc"]
                ),
                "bc": np.sum(dconv, axis=(0, 1)),
            }
            optimizer.step(self.params, grads)
            val_loss = weighted_bce(
                self._forward(x_val)[3],
                y_val,
                pos_weight,
            )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_params = {
                    key: value.copy()
                    for key, value in self.params.items()
                }
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_params is not None:
            self.params = best_params
        return {
            "best_val_bce": best_loss,
            "best_epoch": best_epoch,
            "pos_weight": pos_weight,
        }

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[3]



def sequence_standardization_stats(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    sequences = [
        np.asarray(dataset[rollout_id]["sequence"], dtype=np.float64)
        for rollout_id in rollout_ids
    ]
    if not sequences:
        raise ValueError("empty rollout selection")
    merged = np.concatenate(sequences, axis=0)
    mean = merged.mean(axis=0, keepdims=True)
    std = merged.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class TinyBiLSTM:
    """One-layer bidirectional LSTM over an entire rollout sequence."""

    def __init__(
        self,
        *,
        seed: int,
        epochs: int,
        patience: int,
        hidden: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.epochs = epochs
        self.patience = patience
        self.hidden = hidden
        self.input_dim = 2
        self.params: dict[str, np.ndarray] = {}

    def _init_params(self) -> None:
        joint = self.input_dim + self.hidden
        scale = math.sqrt(1.0 / max(1, joint))
        self.params = {
            "wf": self.rng.normal(
                0.0, scale, size=(joint, 4 * self.hidden)
            ),
            "bf": np.zeros(4 * self.hidden, dtype=np.float64),
            "wb": self.rng.normal(
                0.0, scale, size=(joint, 4 * self.hidden)
            ),
            "bb": np.zeros(4 * self.hidden, dtype=np.float64),
            "wo": self.rng.normal(
                0.0,
                math.sqrt(1.0 / max(1, 2 * self.hidden)),
                size=(2 * self.hidden,),
            ),
            "bo": np.zeros(1, dtype=np.float64),
        }
        # A mildly positive forget bias is the only LSTM-specific
        # stabilization used here; there is no architecture search.
        self.params["bf"][self.hidden : 2 * self.hidden] = 1.0
        self.params["bb"][self.hidden : 2 * self.hidden] = 1.0

    def _direction_forward(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
        h = np.zeros(self.hidden, dtype=np.float64)
        cell = np.zeros(self.hidden, dtype=np.float64)
        outputs = np.zeros((len(x), self.hidden), dtype=np.float64)
        cache: list[tuple[np.ndarray, ...]] = []
        for index in range(len(x)):
            previous_h = h
            previous_cell = cell
            joined = np.concatenate([x[index], previous_h])
            gates = joined @ w + b
            i = _sigmoid(gates[0 : self.hidden])
            f = _sigmoid(gates[self.hidden : 2 * self.hidden])
            o = _sigmoid(gates[2 * self.hidden : 3 * self.hidden])
            g = np.tanh(gates[3 * self.hidden :])
            cell = f * previous_cell + i * g
            h = o * np.tanh(cell)
            outputs[index] = h
            cache.append(
                (joined, i, f, o, g, previous_cell, cell)
            )
        return outputs, cache

    def _direction_backward(
        self,
        dh: np.ndarray,
        cache: Sequence[tuple[np.ndarray, ...]],
        w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        grad_w = np.zeros_like(w)
        grad_b = np.zeros(4 * self.hidden, dtype=np.float64)
        next_h = np.zeros(self.hidden, dtype=np.float64)
        next_cell = np.zeros(self.hidden, dtype=np.float64)
        for index in range(len(cache) - 1, -1, -1):
            joined, i, f, o, g, previous_cell, cell = cache[index]
            total_h = dh[index] + next_h
            tanh_cell = np.tanh(cell)
            d_o = total_h * tanh_cell
            d_cell = (
                total_h * o * (1.0 - tanh_cell * tanh_cell)
                + next_cell
            )
            d_f = d_cell * previous_cell
            d_previous_cell = d_cell * f
            d_i = d_cell * g
            d_g = d_cell * i
            d_gates = np.concatenate(
                [
                    d_i * i * (1.0 - i),
                    d_f * f * (1.0 - f),
                    d_o * o * (1.0 - o),
                    d_g * (1.0 - g * g),
                ]
            )
            grad_w += np.outer(joined, d_gates)
            grad_b += d_gates
            d_joined = w @ d_gates
            next_h = d_joined[self.input_dim :]
            next_cell = d_previous_cell
        return grad_w, grad_b

    def _forward(
        self,
        x: np.ndarray,
        *,
        with_cache: bool,
    ) -> tuple[
        np.ndarray,
        tuple[
            list[tuple[np.ndarray, ...]],
            list[tuple[np.ndarray, ...]],
            np.ndarray,
            np.ndarray,
        ] | None,
    ]:
        forward_h, forward_cache = self._direction_forward(
            x,
            self.params["wf"],
            self.params["bf"],
        )
        reversed_h, backward_cache = self._direction_forward(
            x[::-1],
            self.params["wb"],
            self.params["bb"],
        )
        backward_h = reversed_h[::-1]
        joined_h = np.concatenate([forward_h, backward_h], axis=1)
        logits = joined_h @ self.params["wo"] + self.params["bo"][0]
        if not with_cache:
            return logits, None
        return logits, (
            forward_cache,
            backward_cache,
            forward_h,
            backward_h,
        )

    def _loss(
        self,
        sequences: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        pos_weight: float,
    ) -> float:
        weighted_loss = 0.0
        total_weight = 0.0
        for x, y in zip(sequences, labels):
            logits, _ = self._forward(x, with_cache=False)
            weights = np.where(y > 0.5, pos_weight, 1.0)
            losses = (
                np.maximum(logits, 0.0)
                - logits * y
                + np.log1p(np.exp(-np.abs(logits)))
            )
            weighted_loss += float(np.sum(weights * losses))
            total_weight += float(np.sum(weights))
        return weighted_loss / max(1e-12, total_weight)

    def fit_sequences(
        self,
        train_sequences: Sequence[np.ndarray],
        train_labels: Sequence[np.ndarray],
        val_sequences: Sequence[np.ndarray],
        val_labels: Sequence[np.ndarray],
    ) -> dict[str, Any]:
        self._init_params()
        all_train_labels = np.concatenate(train_labels)
        pos_weight = _pos_weight(all_train_labels)
        train_weight = sum(
            float(np.sum(np.where(y > 0.5, pos_weight, 1.0)))
            for y in train_labels
        )
        optimizer = Adam(self.params, lr=0.003)
        best_loss = math.inf
        best_params = None
        best_epoch = 0
        stale = 0

        for epoch in range(1, self.epochs + 1):
            grads = {
                key: np.zeros_like(value)
                for key, value in self.params.items()
            }
            for x, y in zip(train_sequences, train_labels):
                logits, cache = self._forward(x, with_cache=True)
                assert cache is not None
                forward_cache, backward_cache, forward_h, backward_h = cache
                weights = np.where(y > 0.5, pos_weight, 1.0)
                d_logits = (
                    weights * (_sigmoid(logits) - y)
                    / max(1e-12, train_weight)
                )
                joined_h = np.concatenate(
                    [forward_h, backward_h],
                    axis=1,
                )
                grads["wo"] += joined_h.T @ d_logits
                grads["bo"][0] += float(np.sum(d_logits))

                d_joined = d_logits[:, None] * self.params["wo"][None, :]
                d_forward = d_joined[:, : self.hidden]
                d_backward = d_joined[:, self.hidden :]

                grad_wf, grad_bf = self._direction_backward(
                    d_forward,
                    forward_cache,
                    self.params["wf"],
                )
                # backward_cache is indexed in reversed-time order.
                grad_wb, grad_bb = self._direction_backward(
                    d_backward[::-1],
                    backward_cache,
                    self.params["wb"],
                )
                grads["wf"] += grad_wf
                grads["bf"] += grad_bf
                grads["wb"] += grad_wb
                grads["bb"] += grad_bb

            for key in ("wf", "wb", "wo"):
                grads[key] += 1e-4 * self.params[key]
            grad_norm = math.sqrt(
                sum(float(np.sum(grad * grad)) for grad in grads.values())
            )
            if grad_norm > 5.0:
                scale = 5.0 / grad_norm
                for grad in grads.values():
                    grad *= scale
            optimizer.step(self.params, grads)

            val_loss = self._loss(
                val_sequences,
                val_labels,
                pos_weight,
            )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_params = {
                    key: value.copy()
                    for key, value in self.params.items()
                }
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_params is not None:
            self.params = best_params
        return {
            "best_val_bce": best_loss,
            "best_epoch": best_epoch,
            "pos_weight": pos_weight,
            "hidden": self.hidden,
            "sequence_model": True,
        }

    def logits_sequence(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(x, with_cache=False)
        return logits


def make_sequence_model(
    name: str,
    *,
    seed: int,
    epochs: int,
    patience: int,
) -> TinyBiLSTM:
    hidden_by_name = {
        "tiny_bilstm_h16": 16,
        "tiny_bilstm_h32": 32,
    }
    if name not in hidden_by_name:
        raise ValueError(f"unknown sequence model: {name}")
    return TinyBiLSTM(
        seed=seed,
        epochs=epochs,
        patience=patience,
        hidden=hidden_by_name[name],
    )


def make_model(
    name: str,
    *,
    seed: int,
    epochs: int,
    patience: int,
) -> BaseHead:
    if name == "linear_probe":
        return LinearProbe(seed=seed, epochs=epochs, patience=patience)
    if name == "tiny_mlp":
        return TinyMLP(seed=seed, epochs=epochs, patience=patience)
    if name == "tiny_cnn":
        return TinyCNN(seed=seed, epochs=epochs, patience=patience)
    raise ValueError(f"unknown model: {name}")


def _stratum(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_id: str,
) -> tuple[str, str]:
    event = dataset[rollout_id]["event"]
    task = str(
        event.get("task_key")
        or f"task:{event.get('task_id')}"
    )
    return task, str(event.get("failure_type") or "unknown")


def rollout_split(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rollout_id in sorted(dataset):
        groups[_stratum(dataset, rollout_id)].append(rollout_id)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for key in sorted(groups):
        ids = list(groups[key])
        rng.shuffle(ids)
        count = len(ids)
        if count == 1:
            train.extend(ids)
            continue
        if count == 2:
            train.append(ids[0])
            test.append(ids[1])
            continue
        train_n = max(
            1,
            min(count - 2, int(round(count * train_fraction))),
        )
        val_n = max(
            1,
            min(
                count - train_n - 1,
                int(round(count * val_fraction)),
            ),
        )
        train.extend(ids[:train_n])
        val.extend(ids[train_n : train_n + val_n])
        test.extend(ids[train_n + val_n :])

    all_ids = set(dataset)
    train_set, val_set, test_set = set(train), set(val), set(test)
    train_set.update(all_ids - train_set - val_set - test_set)
    target_min = 1 if len(all_ids) < 12 else 2
    for target in (val_set, test_set):
        while len(target) < target_min and len(train_set) > target_min + 2:
            rollout_id = sorted(train_set)[0]
            train_set.remove(rollout_id)
            target.add(rollout_id)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("rollout split overlap")
    if train_set | val_set | test_set != all_ids:
        raise AssertionError("rollout split does not cover dataset")
    return {
        "split_id": f"random_seed_{seed}",
        "kind": "rollout_random",
        "seed": seed,
        "train": sorted(train_set),
        "val": sorted(val_set),
        "test": sorted(test_set),
    }


def task_splits(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    min_test_rollouts: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for rollout_id, row in dataset.items():
        event = row["event"]
        task = str(
            event.get("task_key")
            or f"task:{event.get('task_id')}"
        )
        by_task[task].append(rollout_id)

    result: list[dict[str, Any]] = []
    for task, test_ids in sorted(by_task.items()):
        if len(test_ids) < min_test_rollouts:
            continue
        remaining = sorted(set(dataset) - set(test_ids))
        if len(remaining) < 5:
            continue
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(remaining)
        val_n = max(
            1,
            min(
                len(remaining) - 2,
                int(round(0.20 * len(remaining))),
            ),
        )
        result.append(
            {
                "split_id": f"task_holdout::{task}",
                "kind": "task_held_out",
                "seed": seed,
                "held_out_task": task,
                "train": sorted(remaining[val_n:]),
                "val": sorted(remaining[:val_n]),
                "test": sorted(test_ids),
            }
        )
    return result


def training_subset(
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    requested: int | None,
    *,
    seed: int,
) -> list[str]:
    if requested is None or requested >= len(train_ids):
        return sorted(train_ids)
    by_task: dict[str, list[str]] = defaultdict(list)
    for rollout_id in train_ids:
        by_task[_stratum(dataset, rollout_id)[0]].append(rollout_id)
    rng = random.Random(seed)
    for ids in by_task.values():
        rng.shuffle(ids)

    result: list[str] = []
    while len(result) < requested:
        moved = False
        for task in sorted(by_task):
            if by_task[task]:
                result.append(by_task[task].pop())
                moved = True
                if len(result) == requested:
                    break
        if not moved:
            break
    return sorted(result)


def interval_error(
    prediction: int,
    causal: int,
    observable: int,
) -> int:
    if prediction < causal:
        return prediction - causal
    if prediction > observable:
        return prediction - observable
    return 0


def prediction_row(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_id: str,
    prediction: int,
    score: float | None,
) -> dict[str, Any]:
    row = dataset[rollout_id]
    error = interval_error(
        prediction,
        int(row["causal_index"]),
        int(row["observable_index"]),
    )
    event = row["event"]
    frames = row["frames"]
    return {
        "rollout_id": rollout_id,
        "event_id": event["event_id"],
        "task_key": event.get("task_key"),
        "task_id": event.get("task_id"),
        "failure_type": event.get("failure_type"),
        "causal_index": int(row["causal_index"]),
        "observable_index": int(row["observable_index"]),
        "predicted_index": int(prediction),
        "predicted_frame": int(frames[prediction]),
        "score": score,
        "interval_error_samples": int(error),
        "in_interval": error == 0,
    }


def metric_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors = [int(row["interval_error_samples"]) for row in rows]
    count = len(errors)
    if not count:
        return {key: None for key in METRIC_NAMES} | {"n": 0}
    absolute = [abs(value) for value in errors]
    result: dict[str, Any] = {
        "n": count,
        "in_interval_rate": (
            sum(value == 0 for value in errors) / count
        ),
        "before_interval_rate": (
            sum(value < 0 for value in errors) / count
        ),
        "after_interval_rate": (
            sum(value > 0 for value in errors) / count
        ),
        "median_absolute_interval_error_samples": float(
            statistics.median(absolute)
        ),
        "mae_samples": float(sum(absolute) / count),
        "mse_samples": float(
            sum(value * value for value in errors) / count
        ),
    }
    for window in (1, 3, 5):
        result[f"within_{window}"] = (
            sum(abs(value) <= window for value in errors) / count
        )
    return result


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_baseline_artifacts(
    run_root: Path,
    explicit: str | None,
) -> Path | None:
    if explicit:
        directory = ensure_within_project(
            resolve_project_path(explicit),
            "baseline analysis dir",
        )
        metadata_path = directory / "metadata.json"
        diagnostics_path = directory / "offline_localization_diagnostics.csv"
        ranking_path = directory / "interval_localization_ranking.csv"
        if not metadata_path.is_file() or not diagnostics_path.is_file() or not ranking_path.is_file():
            raise FileNotFoundError(
                "Explicit baseline analysis must contain metadata.json, "
                "interval_localization_ranking.csv, and "
                "offline_localization_diagnostics.csv"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = project_relative(run_root)
        actual = str((metadata.get("input") or {}).get("run_root") or "")
        if actual != expected:
            raise ValueError(
                "Baseline analysis run_root does not match requested Robo-Dopamine run: "
                f"{actual!r} != {expected!r}"
            )
        return directory
    expected = project_relative(run_root)
    candidates: list[tuple[float, Path]] = []
    if DEFAULT_BASELINE_ROOT.is_dir():
        for metadata_path in DEFAULT_BASELINE_ROOT.glob("*/metadata.json"):
            directory = metadata_path.parent
            if not (
                directory / "interval_localization_ranking.csv"
            ).is_file():
                continue
            if not (
                directory / "offline_localization_diagnostics.csv"
            ).is_file():
                continue
            try:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            input_metadata = metadata.get("input") or {}
            if str(input_metadata.get("run_root") or "") != expected:
                continue
            candidates.append(
                (metadata_path.stat().st_mtime, directory)
            )
    return max(candidates)[1] if candidates else None


def baseline_candidates_from_artifacts(
    directory: Path,
) -> dict[str, Any]:
    return {
        "ranking": _read_csv(
            directory / "interval_localization_ranking.csv"
        ),
        "diagnostics": _read_csv(
            directory / "offline_localization_diagnostics.csv"
        ),
        "source": project_relative(directory),
    }


def _baseline_artifacts_from_compute(
    *,
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fingerprint = search_fingerprint(
        signals,
        events,
        no_event_failures,
        clean_rollouts,
    )
    cached = load_search_cache(fingerprint)
    source = "search_cache"
    if cached is None:
        source = "computed_existing_sweep"
        configs, _oracle, _grid = build_phenotype_detector_configs(
            signals,
            events,
            no_event_failures,
            clean_rollouts,
        )
        (
            _summary,
            event_rows,
            no_event_rows,
            clean_rows,
        ) = evaluate_all_configs(
            configs,
            signals,
            events,
            no_event_failures,
            clean_rollouts,
        )
        ensemble_sweep, _selected, _failure = (
            build_pairwise_ensemble_rows(
                configs,
                event_rows,
                no_event_rows,
                clean_rows,
            )
        )
    else:
        configs = list(cached["configs"])
        event_rows = list(cached["event_rows"])
        ensemble_sweep = list(cached["ensemble_sweep"])

    ranking, diagnostics = offline_change_point_localization(
        event_rows=event_rows,
        ensemble_sweep=ensemble_sweep,
        signals=signals,
        configs=configs,
    )
    return {
        "ranking": ranking,
        "diagnostics": diagnostics,
        "source": source,
    }


def _as_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def build_baseline_prediction_bank(
    baseline: Mapping[str, Any],
    primary_event_ids: Mapping[str, str],
) -> dict[str, dict[tuple[Any, ...], dict[str, int]]]:
    first: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    offline: dict[tuple[Any, ...], dict[str, int]] = defaultdict(dict)
    seen_first: set[tuple[str, str]] = set()
    for row in baseline["diagnostics"]:
        rollout_id = str(row.get("rollout_id") or "")
        if (
            not rollout_id
            or str(row.get("event_id") or "")
            != primary_event_ids.get(rollout_id)
        ):
            continue
        config_id = str(row.get("config_id") or "")
        first_index = _as_int(
            row.get("first_trigger_sample_index")
        )
        if (
            first_index is not None
            and (config_id, rollout_id) not in seen_first
        ):
            first[(config_id,)][rollout_id] = first_index
            seen_first.add((config_id, rollout_id))

        selected = _as_int(
            row.get("selected_trigger_sample_index")
        )
        left = _as_int(row.get("L"))
        right = _as_int(row.get("R"))
        if (
            selected is not None
            and left is not None
            and right is not None
        ):
            offline[(config_id, left, right)][
                rollout_id
            ] = selected
    return {
        "first_trigger": first,
        "offline_changepoint": offline,
    }


def choose_baseline_candidate(
    bank: Mapping[tuple[Any, ...], Mapping[str, int]],
    dataset: Mapping[str, Mapping[str, Any]],
    val_ids: Sequence[str],
) -> tuple[tuple[Any, ...] | None, dict[str, Any]]:
    scored = []
    for key, predictions in bank.items():
        rows = [
            prediction_row(
                dataset,
                rollout_id,
                int(predictions[rollout_id]),
                None,
            )
            for rollout_id in val_ids
            if rollout_id in predictions
        ]
        if not rows:
            continue
        metrics = metric_summary(rows)
        coverage = (
            len(rows) / len(val_ids)
            if val_ids
            else 0.0
        )
        scored.append(
            (
                metrics["mse_samples"],
                metrics["mae_samples"],
                -metrics["in_interval_rate"],
                -coverage,
                key,
                metrics | {"coverage": coverage},
            )
        )
    if not scored:
        return None, {"coverage": 0.0}
    max_coverage = max(float(item[5]["coverage"]) for item in scored)
    coverage_floor = (
        1.0
        if any(float(item[5]["coverage"]) >= 1.0 - 1e-12 for item in scored)
        else max_coverage
    )
    eligible = [
        item
        for item in scored
        if float(item[5]["coverage"]) >= coverage_floor - 1e-12
    ]
    eligible.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[4],
        )
    )
    winner = eligible[0]
    winner[5]["coverage_floor"] = coverage_floor
    return winner[4], winner[5]


def evaluate_fixed_predictions(
    name: str,
    predictions: Mapping[str, int],
    dataset: Mapping[str, Mapping[str, Any]],
    test_ids: Sequence[str],
    *,
    split: Mapping[str, Any],
    train_size_label: str,
    candidate_key: tuple[Any, ...] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for rollout_id in test_ids:
        if rollout_id not in predictions:
            continue
        row = prediction_row(
            dataset,
            rollout_id,
            int(predictions[rollout_id]),
            None,
        )
        row.update(
            {
                "split_kind": split["kind"],
                "split_id": split["split_id"],
                "method": name,
                "train_size": train_size_label,
                "candidate_key": (
                    json.dumps(candidate_key)
                    if candidate_key is not None
                    else None
                ),
            }
        )
        rows.append(row)

    metrics = metric_summary(rows)
    metrics.update(
        {
            "split_kind": split["kind"],
            "split_id": split["split_id"],
            "held_out_task": split.get("held_out_task"),
            "method": name,
            "train_size": train_size_label,
            "test_rollout_n": len(test_ids),
            "prediction_coverage": (
                len(rows) / len(test_ids)
                if test_ids
                else None
            ),
            "candidate_key": (
                json.dumps(candidate_key)
                if candidate_key is not None
                else None
            ),
        }
    )
    return metrics, rows


def train_and_evaluate_model(
    name: str,
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    seed: int,
    epochs: int,
    patience: int,
    split: Mapping[str, Any],
    train_size_label: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    x_train, y_train = flatten_dataset(dataset, train_ids)
    x_val, y_val = flatten_dataset(dataset, val_ids)
    mean, std = standardization_stats(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    model = make_model(
        name,
        seed=seed,
        epochs=epochs,
        patience=patience,
    )
    training = model.fit(
        x_train,
        y_train,
        x_val,
        y_val,
    )

    predictions: list[dict[str, Any]] = []
    for rollout_id in test_ids:
        x = (
            np.asarray(
                dataset[rollout_id]["context"],
                dtype=np.float64,
            )
            - mean
        ) / std
        logits = model.logits(x)
        prediction = int(np.argmax(logits))
        row = prediction_row(
            dataset,
            rollout_id,
            prediction,
            float(logits[prediction]),
        )
        row.update(
            {
                "split_kind": split["kind"],
                "split_id": split["split_id"],
                "method": name,
                "train_size": train_size_label,
                "candidate_key": None,
            }
        )
        predictions.append(row)

    metrics = metric_summary(predictions)
    metrics.update(
        {
            "split_kind": split["kind"],
            "split_id": split["split_id"],
            "held_out_task": split.get("held_out_task"),
            "method": name,
            "train_size": train_size_label,
            "train_rollout_n": len(train_ids),
            "val_rollout_n": len(val_ids),
            "test_rollout_n": len(test_ids),
            "prediction_coverage": 1.0,
            "candidate_key": None,
            "best_epoch": training["best_epoch"],
            "best_val_bce": training["best_val_bce"],
            "pos_weight": training["pos_weight"],
        }
    )
    return metrics, predictions, training



def train_and_evaluate_sequence_model(
    name: str,
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    seed: int,
    epochs: int,
    patience: int,
    split: Mapping[str, Any],
    train_size_label: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    mean, std = sequence_standardization_stats(dataset, train_ids)

    def normalized(ids: Sequence[str]) -> list[np.ndarray]:
        return [
            (
                np.asarray(dataset[rollout_id]["sequence"], dtype=np.float64)
                - mean
            ) / std
            for rollout_id in ids
        ]

    train_sequences = normalized(train_ids)
    val_sequences = normalized(val_ids)
    train_labels = [
        np.asarray(dataset[rollout_id]["labels"], dtype=np.float64)
        for rollout_id in train_ids
    ]
    val_labels = [
        np.asarray(dataset[rollout_id]["labels"], dtype=np.float64)
        for rollout_id in val_ids
    ]

    model = make_sequence_model(
        name,
        seed=seed,
        epochs=epochs,
        patience=patience,
    )
    training = model.fit_sequences(
        train_sequences,
        train_labels,
        val_sequences,
        val_labels,
    )

    predictions: list[dict[str, Any]] = []
    for rollout_id in test_ids:
        sequence = (
            np.asarray(dataset[rollout_id]["sequence"], dtype=np.float64)
            - mean
        ) / std
        logits = model.logits_sequence(sequence)
        prediction = int(np.argmax(logits))
        row = prediction_row(
            dataset,
            rollout_id,
            prediction,
            float(logits[prediction]),
        )
        row.update(
            {
                "split_kind": split["kind"],
                "split_id": split["split_id"],
                "method": name,
                "train_size": train_size_label,
                "candidate_key": None,
            }
        )
        predictions.append(row)

    metrics = metric_summary(predictions)
    metrics.update(
        {
            "split_kind": split["kind"],
            "split_id": split["split_id"],
            "held_out_task": split.get("held_out_task"),
            "method": name,
            "train_size": train_size_label,
            "train_rollout_n": len(train_ids),
            "val_rollout_n": len(val_ids),
            "test_rollout_n": len(test_ids),
            "prediction_coverage": 1.0,
            "candidate_key": None,
            "best_epoch": training["best_epoch"],
            "best_val_bce": training["best_val_bce"],
            "pos_weight": training["pos_weight"],
        }
    )
    return metrics, predictions, training


def train_and_evaluate_any_model(
    name: str,
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if name in SEQUENCE_MODEL_NAMES:
        return train_and_evaluate_sequence_model(
            name,
            dataset,
            train_ids,
            val_ids,
            test_ids,
            **kwargs,
        )
    return train_and_evaluate_model(
        name,
        dataset,
        train_ids,
        val_ids,
        test_ids,
        **kwargs,
    )


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[Any, ...],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[
            tuple(row.get(field) for field in group_fields)
        ].append(row)

    result = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        out = {
            field: value
            for field, value in zip(group_fields, key)
        }
        out["repeat_n"] = len(group)
        for metric in METRIC_NAMES:
            values = [
                float(row[metric])
                for row in group
                if row.get(metric) not in (None, "")
            ]
            out[f"{metric}_mean"] = (
                float(np.mean(values))
                if values
                else None
            )
            out[f"{metric}_variance"] = (
                float(np.var(values))
                if values
                else None
            )
        coverages = [
            float(row["prediction_coverage"])
            for row in group
            if row.get("prediction_coverage") is not None
        ]
        out["prediction_coverage_mean"] = (
            float(np.mean(coverages))
            if coverages
            else None
        )
        train_ns = [
            float(row["train_rollout_n"])
            for row in group
            if row.get("train_rollout_n") is not None
        ]
        out["train_rollout_n_mean"] = (
            float(np.mean(train_ns))
            if train_ns
            else None
        )
        result.append(out)
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def conclusion_text(
    model_comparison: Sequence[Mapping[str, Any]],
    learning_curve: Sequence[Mapping[str, Any]],
) -> str:
    learned = [
        row
        for row in model_comparison
        if row.get("method") in MODEL_NAMES
    ]
    baselines = [
        row
        for row in model_comparison
        if row.get("method") not in MODEL_NAMES
    ]
    if not learned:
        return "No learned-head result was produced.\n"

    best_learned = min(
        learned,
        key=lambda row: (
            float(row.get("mae_samples_mean") or math.inf),
            -float(row.get("in_interval_rate_mean") or 0.0),
        ),
    )
    best_baseline = min(
        baselines,
        key=lambda row: (
            float(row.get("mae_samples_mean") or math.inf),
            -float(row.get("in_interval_rate_mean") or 0.0),
        ),
        default=None,
    )

    clearly_better = False
    if best_baseline is not None:
        clearly_better = (
            float(best_learned.get("mae_samples_mean") or math.inf)
            < float(best_baseline.get("mae_samples_mean") or math.inf)
            and float(
                best_learned.get("in_interval_rate_mean") or 0.0
            )
            > float(
                best_baseline.get("in_interval_rate_mean") or 0.0
            )
        )

    curve = [
        row
        for row in learning_curve
        if row.get("method") == best_learned.get("method")
    ]
    curve_by_size = {
        str(row.get("train_size")): row
        for row in curve
    }
    all_row = curve_by_size.get("all")
    finite_rows = [
        row
        for row in curve
        if str(row.get("train_size")) != "all"
    ]
    largest_fixed = max(
        finite_rows,
        key=lambda row: float(
            row.get("train_rollout_n_mean") or 0.0
        ),
        default=None,
    )
    data_limited = None
    if all_row is not None and largest_fixed is not None:
        fixed_mae = float(
            largest_fixed.get("mae_samples_mean") or math.inf
        )
        all_mae = float(
            all_row.get("mae_samples_mean") or math.inf
        )
        fixed_hit = float(
            largest_fixed.get("in_interval_rate_mean") or 0.0
        )
        all_hit = float(
            all_row.get("in_interval_rate_mean") or 0.0
        )
        data_limited = (
            all_mae < 0.90 * fixed_mae
            or all_hit > fixed_hit + 0.05
        )

    lines = [
        "# Lightweight Robo-Dopamine localization-head conclusion",
        "",
        (
            "Best learned head by mean MAE: "
            f"**{best_learned['method']}**."
        ),
    ]
    if best_baseline is not None:
        lines.append(
            "Best baseline by the same criterion: "
            f"**{best_baseline['method']}**."
        )
        if clearly_better:
            lines.append(
                "The learned head improves both mean MAE and mean "
                "in-interval rate over the strongest baseline across "
                "the repeated rollout splits. This supports continuing "
                "the failure-localization-head direction, while the "
                "task-held-out results should still be checked before "
                "moving to richer GRM features."
            )
        else:
            lines.append(
                "The learned head does not simultaneously improve mean "
                "MAE and mean in-interval rate over the strongest "
                "baseline. On this probe, richer Robo-Dopamine hidden "
                "features are not yet justified by the temporal-signal "
                "evidence alone."
            )
    if data_limited is True:
        lines.append(
            "The learning curve still improves materially from the "
            "largest fixed-size subset to all training rollouts, so "
            "current performance appears data-limited; scaling toward "
            "roughly 100–200 annotated failures is worth testing."
        )
    elif data_limited is False:
        lines.append(
            "The learning curve changes little from the largest "
            "fixed-size subset to all training rollouts, suggesting "
            "the current temporal features/model family may be closer "
            "to saturation than simply data-limited."
        )
    else:
        lines.append(
            "The available split sizes were insufficient to make a "
            "reliable saturation-versus-data-limited call."
        )
    lines.append(
        "The probe now includes tiny one-layer BiLSTMs with hidden sizes "
        "16 and 32. These consume the full rollout sequence rather than "
        "the local symmetric window, directly testing whether longer "
        "offline temporal context improves changepoint localization."
    )
    return "\n\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> Path:
    run_root = ensure_within_project(
        resolve_project_path(args.run_root),
        "run root",
    )
    manifest_path = ensure_within_project(
        resolve_project_path(args.manifest),
        "manifest",
    )
    annotation_dir = ensure_within_project(
        resolve_project_path(args.annotations),
        "annotation directory",
    )
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not annotation_dir.is_dir():
        raise FileNotFoundError(annotation_dir)

    if args.output_dir:
        output_dir = ensure_within_project(
            resolve_project_path(args.output_dir),
            "output directory",
        )
    else:
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    (
        signals,
        events,
        no_event_failures,
        clean_rollouts,
        provenance,
    ) = build_base_records(
        run_root,
        manifest,
        annotation_dir,
        allowed_rollout_ids=None,
        signal_mode="fused",
    )
    dataset = build_dataset(
        signals,
        events,
        args.context_radius,
    )
    if len(dataset) < 8:
        raise ValueError(
            "Need at least 8 eligible terminal-failure rollouts, "
            f"found {len(dataset)}"
        )

    random_splits = [
        rollout_split(
            dataset,
            seed=args.seed + repeat,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        for repeat in range(args.repeats)
    ]
    heldout_splits = task_splits(
        dataset,
        seed=args.seed,
        min_test_rollouts=args.min_task_test_rollouts,
    )

    baseline_dir = discover_baseline_artifacts(
        run_root,
        args.baseline_analysis_dir,
    )
    if baseline_dir is not None:
        baseline = baseline_candidates_from_artifacts(
            baseline_dir
        )
    else:
        baseline = _baseline_artifacts_from_compute(
            signals=signals,
            events=events,
            no_event_failures=no_event_failures,
            clean_rollouts=clean_rollouts,
        )

    primary_event_ids = {
        rollout_id: str(row["event"]["event_id"])
        for rollout_id, row in dataset.items()
    }
    banks = build_baseline_prediction_bank(
        baseline,
        primary_event_ids,
    )

    split_manifest = {
        "schema_version": 1,
        "target": (
            "first eligible terminal-failure event per rollout"
        ),
        "run_root": project_relative(run_root),
        "context": {
            "signals": ["fused_progress", "fused_hop"],
            "radius": args.context_radius,
            "width": 2 * args.context_radius + 1,
            "mode": "symmetric offline context",
        },
        "rollouts": [
            {
                "rollout_id": rollout_id,
                "event_id": row["event"]["event_id"],
                "task_key": row["event"].get("task_key"),
                "task_id": row["event"].get("task_id"),
                "failure_type": row["event"].get("failure_type"),
                "causal_native_index": int(
                    row["causal_index"]
                ),
                "observable_native_index": int(
                    row["observable_index"]
                ),
                "native_sample_n": int(len(row["labels"])),
            }
            for rollout_id, row in sorted(dataset.items())
        ],
        "random_splits": random_splits,
        "task_held_out_splits": heldout_splits,
    }
    write_json(
        output_dir / "split_manifest.json",
        split_manifest,
    )

    detailed_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []

    for split in random_splits:
        requested_sizes: list[int | None] = [
            10,
            20,
            30,
            None,
        ]
        for requested in requested_sizes:
            if (
                requested is not None
                and requested > len(split["train"])
            ):
                continue
            train_ids = training_subset(
                dataset,
                split["train"],
                requested,
                seed=int(split["seed"]) * 100 + 7,
            )
            label = (
                "all"
                if requested is None
                else str(requested)
            )
            for model_offset, model_name in enumerate(
                MODEL_NAMES
            ):
                metrics, rows, training = (
                    train_and_evaluate_any_model(
                        model_name,
                        dataset,
                        train_ids,
                        split["val"],
                        split["test"],
                        seed=(
                            int(split["seed"]) * 1000
                            + model_offset
                        ),
                        epochs=args.epochs,
                        patience=args.patience,
                        split=split,
                        train_size_label=label,
                    )
                )
                detailed_metrics.append(metrics)
                predictions.extend(rows)
                training_records.append(
                    {
                        "split_id": split["split_id"],
                        "model": model_name,
                        "train_size": label,
                        "train_rollout_ids": train_ids,
                        **training,
                    }
                )

        for baseline_name, bank_name in (
            (
                "handcrafted_first_trigger",
                "first_trigger",
            ),
            (
                "offline_changepoint",
                "offline_changepoint",
            ),
        ):
            candidate, val_metrics = (
                choose_baseline_candidate(
                    banks[bank_name],
                    dataset,
                    split["val"],
                )
            )
            pred_bank = (
                banks[bank_name].get(candidate, {})
                if candidate is not None
                else {}
            )
            metrics, rows = evaluate_fixed_predictions(
                baseline_name,
                pred_bank,
                dataset,
                split["test"],
                split=split,
                train_size_label="all",
                candidate_key=candidate,
            )
            metrics["train_rollout_n"] = len(
                split["train"]
            )
            metrics["val_selection_metrics"] = json.dumps(
                val_metrics,
                sort_keys=True,
            )
            detailed_metrics.append(metrics)
            predictions.extend(rows)

        global_predictions = {
            rollout_id: int(
                np.argmax(dataset[rollout_id]["progress"])
            )
            for rollout_id in dataset
        }
        metrics, rows = evaluate_fixed_predictions(
            "global_fused_progress_argmax",
            global_predictions,
            dataset,
            split["test"],
            split=split,
            train_size_label="all",
        )
        metrics["train_rollout_n"] = len(split["train"])
        detailed_metrics.append(metrics)
        predictions.extend(rows)

    task_metrics: list[dict[str, Any]] = []
    for split_index, split in enumerate(heldout_splits):
        for model_offset, model_name in enumerate(
            MODEL_NAMES
        ):
            metrics, rows, training = (
                train_and_evaluate_any_model(
                    model_name,
                    dataset,
                    split["train"],
                    split["val"],
                    split["test"],
                    seed=(
                        args.seed * 10000
                        + split_index * 100
                        + model_offset
                    ),
                    epochs=args.epochs,
                    patience=args.patience,
                    split=split,
                    train_size_label="all",
                )
            )
            task_metrics.append(metrics)
            predictions.extend(rows)
            training_records.append(
                {
                    "split_id": split["split_id"],
                    "model": model_name,
                    "train_size": "all",
                    "train_rollout_ids": split["train"],
                    **training,
                }
            )

        for baseline_name, bank_name in (
            (
                "handcrafted_first_trigger",
                "first_trigger",
            ),
            (
                "offline_changepoint",
                "offline_changepoint",
            ),
        ):
            candidate, val_metrics = (
                choose_baseline_candidate(
                    banks[bank_name],
                    dataset,
                    split["val"],
                )
            )
            pred_bank = (
                banks[bank_name].get(candidate, {})
                if candidate is not None
                else {}
            )
            metrics, rows = evaluate_fixed_predictions(
                baseline_name,
                pred_bank,
                dataset,
                split["test"],
                split=split,
                train_size_label="all",
                candidate_key=candidate,
            )
            metrics["train_rollout_n"] = len(
                split["train"]
            )
            metrics["val_selection_metrics"] = json.dumps(
                val_metrics,
                sort_keys=True,
            )
            task_metrics.append(metrics)
            predictions.extend(rows)

        global_predictions = {
            rollout_id: int(
                np.argmax(dataset[rollout_id]["progress"])
            )
            for rollout_id in dataset
        }
        metrics, rows = evaluate_fixed_predictions(
            "global_fused_progress_argmax",
            global_predictions,
            dataset,
            split["test"],
            split=split,
            train_size_label="all",
        )
        metrics["train_rollout_n"] = len(split["train"])
        task_metrics.append(metrics)
        predictions.extend(rows)

    full_rows = [
        row
        for row in detailed_metrics
        if row.get("train_size") == "all"
    ]
    model_comparison = aggregate_rows(
        full_rows,
        ["method"],
    )
    learning_curve = aggregate_rows(
        [
            row
            for row in detailed_metrics
            if row.get("method") in MODEL_NAMES
        ],
        ["method", "train_size"],
    )
    task_summary = (
        aggregate_rows(task_metrics, ["method"])
        if task_metrics
        else []
    )

    write_csv(
        output_dir / "model_comparison.csv",
        model_comparison,
    )
    write_csv(
        output_dir / "learning_curve.csv",
        learning_curve,
    )
    write_csv(
        output_dir / "per_split_metrics.csv",
        detailed_metrics,
    )
    write_csv(
        output_dir / "task_held_out_metrics.csv",
        task_metrics,
    )
    write_csv(
        output_dir / "task_held_out_summary.csv",
        task_summary,
    )
    write_csv(
        output_dir / "per_rollout_predictions.csv",
        predictions,
    )
    write_json(
        output_dir / "training_records.json",
        training_records,
    )
    (output_dir / "conclusion.md").write_text(
        conclusion_text(
            model_comparison,
            learning_curve,
        ),
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "analysis": (
            "robo_dopamine_lightweight_localization_head"
        ),
        "analysis_mode": (
            "CPU-only saved-output training; "
            "no Robo-Dopamine inference"
        ),
        "run_root": project_relative(run_root),
        "manifest": project_relative(manifest_path),
        "annotation_dir": project_relative(
            annotation_dir
        ),
        "eligible_rollout_n": len(dataset),
        "target": (
            "first eligible terminal-failure "
            "[causal, observable] interval per rollout"
        ),
        "models": {
            "linear_probe": (
                "weighted logistic regression over "
                "flattened local context"
            ),
            "tiny_mlp": (
                "1-hidden-layer ReLU MLP, hidden=16"
            ),
            "tiny_cnn": (
                "1D conv over local progress/hop context, "
                "filters=8, kernel=3, position-aware readout"
            ),
            "tiny_bilstm_h16": (
                "1-layer bidirectional LSTM over full rollout sequence, "
                "hidden=16 per direction"
            ),
            "tiny_bilstm_h32": (
                "1-layer bidirectional LSTM over full rollout sequence, "
                "hidden=32 per direction"
            ),
        },
        "features": {
            "channels": [
                "fused_progress",
                "fused_hop",
            ],
            "context_radius": args.context_radius,
            "context_mode": (
                "local symmetric windows for linear/MLP/CNN; "
                "full normalized rollout sequence for BiLSTM"
            ),
            "standardization": (
                "train-rollout timesteps only, "
                "per signal channel"
            ),
        },
        "training": {
            "epochs_max": args.epochs,
            "patience": args.patience,
            "loss": (
                "positive-class-weighted BCE"
            ),
            "optimizer": (
                "Adam implemented in NumPy; BiLSTM uses full-sequence "
                "BPTT with global-norm gradient clipping at 5"
            ),
            "random_split_repeats": args.repeats,
            "learning_curve_requested": [
                10,
                20,
                30,
                "all",
            ],
            "strict_rollout_split": True,
            "timestep_split": False,
        },
        "task_held_out": {
            "enabled": bool(heldout_splits),
            "split_n": len(heldout_splits),
            "minimum_test_rollouts_per_task": (
                args.min_task_test_rollouts
            ),
        },
        "baseline_source": baseline.get("source"),
        "baseline_selection": (
            "existing candidate rules selected on "
            "validation rollouts only; test rollouts untouched"
        ),
        "signal_provenance": provenance,
        "outputs": [
            "model_comparison.csv",
            "learning_curve.csv",
            "per_rollout_predictions.csv",
            "split_manifest.json",
            "per_split_metrics.csv",
            "task_held_out_metrics.csv",
            "task_held_out_summary.csv",
            "training_records.json",
            "conclusion.md",
            "metadata.json",
        ],
    }
    write_json(
        output_dir / "metadata.json",
        metadata,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help=(
            "Completed Robo-Dopamine run with "
            "saved fused progress/hop"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--baseline-analysis-dir",
        default=None,
        help=(
            "Optional completed fused-hop analysis "
            "containing offline diagnostics"
        ),
    )
    parser.add_argument(
        "--context-radius",
        type=int,
        default=3,
        help=(
            "Native samples on each side of t; "
            "default 3 => width 7"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--min-task-test-rollouts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=35,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.context_radius < 1:
        parser.error("--context-radius must be >=1")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.epochs < 1 or args.patience < 1:
        parser.error(
            "--epochs and --patience must be positive"
        )
    if (
        not 0 < args.train_fraction < 1
        or not 0 < args.val_fraction < 1
    ):
        parser.error(
            "train/val fractions must be in (0,1)"
        )
    if (
        args.train_fraction
        + args.val_fraction
        >= 1
    ):
        parser.error(
            "train + val fractions must be < 1"
        )
    try:
        output = analyze(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
