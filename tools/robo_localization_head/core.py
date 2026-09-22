"""Shared training core for Robo-Dopamine BiLSTM localization experiments.

Experiment files own target construction, loss definitions, and reporting.
This module owns the common model, rollout split, normalization, variable-length
mini-batching, optimization loop, and batched inference.
"""

from __future__ import annotations

import math
import random
import threading
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

RowLossFn = Callable[
    [torch.Tensor, Mapping[str, Any], torch.Tensor, torch.Tensor],
    torch.Tensor,
]

_SEED_LOCK = threading.Lock()


def onset_anchor(frames: Sequence[int], frame: int) -> int | None:
    return next(
        (index for index, value in enumerate(frames) if int(value) >= int(frame)),
        None,
    )


def sequence_from_signal(signal: Mapping[str, Any]) -> np.ndarray:
    progress = np.asarray(signal["progress"], dtype=np.float32)
    hops = np.asarray(signal["hops"], dtype=np.float32)
    if progress.ndim != 1 or hops.ndim != 1 or len(progress) != len(hops):
        raise ValueError("progress/hops must be same-length 1D arrays")
    if len(progress) == 0:
        raise ValueError("empty fused signal")
    return np.stack([progress, hops], axis=1)


def task_key(dataset: Mapping[str, Mapping[str, Any]], rollout_id: str) -> str:
    return str(dataset[rollout_id].get("task_key") or "")


def rollout_split(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for rollout_id in sorted(dataset):
        by_task[task_key(dataset, rollout_id)].append(rollout_id)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for task in sorted(by_task):
        ids = list(by_task[task])
        rng.shuffle(ids)
        count = len(ids)
        if count == 1:
            train.extend(ids)
            continue
        if count == 2:
            train.append(ids[0])
            test.append(ids[1])
            continue
        train_n = max(1, min(count - 2, int(round(count * train_fraction))))
        val_n = max(1, min(count - train_n - 1, int(round(count * val_fraction))))
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


def force_train_rollouts(
    split: Mapping[str, Any],
    forced_ids: Sequence[str],
) -> dict[str, Any]:
    """Move selected rollouts into train while keeping val/test sizes when possible.

    This is intended for diagnostic challenge-set experiments, not held-out
    generalization estimates. Rollouts moved out of val/test are replaced by
    deterministic non-forced train rollouts when enough candidates exist.
    """
    result = dict(split)
    train = set(split.get("train", []))
    val = set(split.get("val", []))
    test = set(split.get("test", []))
    all_ids = train | val | test
    forced = set(forced_ids) & all_ids
    target_val_n = len(val)
    target_test_n = len(test)

    val.difference_update(forced)
    test.difference_update(forced)
    train.update(forced)

    candidates = sorted(train - forced)
    rng = random.Random(int(split.get("seed", 0)) + 104729)
    rng.shuffle(candidates)

    def refill(target: set[str], target_n: int) -> None:
        while len(target) < target_n and candidates:
            rollout_id = candidates.pop()
            if rollout_id not in train:
                continue
            train.remove(rollout_id)
            target.add(rollout_id)

    refill(val, target_val_n)
    refill(test, target_test_n)

    if not train:
        raise ValueError("forcing challenge rollouts into train left training empty")
    if not val:
        raise ValueError(
            "forcing challenge rollouts into train left validation empty; "
            "reduce the forced challenge set or add more failure rollouts"
        )

    result["train"] = sorted(train)
    result["val"] = sorted(val)
    result["test"] = sorted(test)
    result["forced_train"] = sorted(forced)
    result["kind"] = (
        "rollout_random_force_train" if forced else str(split.get("kind") or "rollout_random")
    )
    return result


def standardization_stats(
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    sequences = [
        np.asarray(dataset[rollout_id]["sequence"], dtype=np.float32)
        for rollout_id in rollout_ids
    ]
    if not sequences:
        raise ValueError("empty rollout selection")
    merged = np.concatenate(sequences, axis=0)
    mean = merged.mean(axis=0, keepdims=True)
    std = merged.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class TinyBiLSTM(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            encoded, _ = self.lstm(sequence)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_encoded, _ = self.lstm(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_encoded,
                batch_first=True,
                total_length=sequence.shape[1],
            )
        return self.head(encoded).squeeze(-1)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_tensor_dataset(
    dataset: Mapping[str, Mapping[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, dict[str, torch.Tensor]]:
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for rollout_id, row in dataset.items():
        sequence = (np.asarray(row["sequence"], dtype=np.float32) - mean) / std
        labels = np.asarray(row["labels"], dtype=np.float32)
        prepared[rollout_id] = {
            "sequence": torch.from_numpy(sequence),
            "labels": torch.from_numpy(labels),
        }
    return prepared


def collate_rollout_batch(
    prepared: Mapping[str, Mapping[str, torch.Tensor]],
    rollout_ids: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rollout_ids:
        raise ValueError("cannot collate an empty rollout batch")
    sequences = [prepared[rollout_id]["sequence"] for rollout_id in rollout_ids]
    labels = [prepared[rollout_id]["labels"] for rollout_id in rollout_ids]
    lengths = torch.tensor(
        [int(sequence.shape[0]) for sequence in sequences],
        dtype=torch.long,
    )
    padded_sequence = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0.0
    )
    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0.0
    )
    non_blocking = device.type == "cuda"
    return (
        padded_sequence.to(device, non_blocking=non_blocking),
        padded_labels.to(device, non_blocking=non_blocking),
        lengths,
    )


def rollout_batches(
    rollout_ids: Sequence[str],
    batch_size: int,
    *,
    rng: random.Random | None = None,
) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ids = list(rollout_ids)
    if rng is not None:
        rng.shuffle(ids)
    return [ids[index : index + batch_size] for index in range(0, len(ids), batch_size)]


def train_bilstm(
    *,
    dataset: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    hidden: int,
    pos_weight: float,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    batch_size: int,
    device: torch.device,
    row_loss_fn: RowLossFn,
    progress_label: str,
    logger: Callable[[str], None],
) -> tuple[TinyBiLSTM, dict[str, Any]]:
    if not train_ids:
        raise ValueError("training split is empty")
    if not val_ids:
        raise ValueError("validation split is empty")

    prepared = prepare_tensor_dataset(dataset, mean, std)
    # Model initialization uses global RNG state. Keep it deterministic when
    # several independent configurations train concurrently in worker threads.
    with _SEED_LOCK:
        set_seed(seed)
        model = TinyBiLSTM(hidden=hidden).to(device)
    weight_params = [
        parameter for name, parameter in model.named_parameters() if "bias" not in name
    ]
    bias_params = [
        parameter for name, parameter in model.named_parameters() if "bias" in name
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "weight_decay": weight_decay},
            {"params": bias_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    effective_train_batch = min(batch_size, len(train_ids))
    effective_val_batch = min(batch_size, len(val_ids))
    optimizer_steps_per_epoch = math.ceil(len(train_ids) / effective_train_batch)
    shuffle_rng = random.Random(seed + 7919)

    def batch_losses(
        batch_ids: Sequence[str],
        logits: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        values: list[torch.Tensor] = []
        for batch_index, rollout_id in enumerate(batch_ids):
            length = int(lengths[batch_index].item())
            values.append(
                row_loss_fn(
                    logits[batch_index, :length],
                    dataset[rollout_id],
                    labels[batch_index, :length],
                    pos_weight_tensor,
                )
            )
        return values

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for batch_ids in rollout_batches(
            train_ids, effective_train_batch, rng=shuffle_rng
        ):
            optimizer.zero_grad(set_to_none=True)
            x, y, lengths = collate_rollout_batch(prepared, batch_ids, device)
            logits = model(x, lengths)
            row_losses = batch_losses(batch_ids, logits, y, lengths)
            row_loss_tensor = torch.stack(row_losses)
            batch_loss = row_loss_tensor.mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # One host synchronization per batch instead of one per rollout.
            train_loss_sum += float(row_loss_tensor.detach().sum().item())
            train_seen += len(row_losses)

        train_loss = train_loss_sum / max(1, train_seen)

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch_ids in rollout_batches(val_ids, effective_val_batch):
                x, y, lengths = collate_rollout_batch(prepared, batch_ids, device)
                logits = model(x, lengths)
                row_losses = batch_losses(batch_ids, logits, y, lengths)
                row_loss_tensor = torch.stack(row_losses)
                # Again synchronize once per batch, not once per rollout.
                val_loss_sum += float(row_loss_tensor.detach().sum().item())
                val_seen += len(row_losses)
        val_loss = val_loss_sum / val_seen if val_seen else float("inf")
        if epoch == 1 or epoch % 25 == 0:
            logger(
                f"{progress_label} epoch={epoch}/{epochs} "
                f"batch={effective_train_batch} steps={optimizer_steps_per_epoch} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best={best_loss:.6f}"
            )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger(
                    f"{progress_label} early_stop epoch={epoch} "
                    f"best_epoch={best_epoch} best_val_loss={best_loss:.6f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "pos_weight": pos_weight,
        "hidden": hidden,
        "device": str(device),
        "batch_size": batch_size,
        "effective_train_batch_size": effective_train_batch,
        "effective_val_batch_size": effective_val_batch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "train_n": len(train_ids),
        "val_n": len(val_ids),
    }


def batched_logits(
    model: TinyBiLSTM,
    dataset: Mapping[str, Mapping[str, Any]],
    rollout_ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    prepared = prepare_tensor_dataset(dataset, mean, std)
    result: dict[str, torch.Tensor] = {}
    effective_batch = min(batch_size, max(1, len(rollout_ids)))
    model.eval()
    with torch.no_grad():
        for batch_ids in rollout_batches(rollout_ids, effective_batch):
            x, _labels, lengths = collate_rollout_batch(prepared, batch_ids, device)
            logits = model(x, lengths)
            for batch_index, rollout_id in enumerate(batch_ids):
                length = int(lengths[batch_index].item())
                result[rollout_id] = logits[batch_index, :length].detach().cpu()
    return result
