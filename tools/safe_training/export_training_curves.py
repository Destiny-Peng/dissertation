#!/usr/bin/env python3
"""Export loss histories from SAFE's local W&B offline runs.

This reader never calls the W&B API.  It scans the local ``.wandb`` files,
writes a tidy CSV, and renders PNG/SVG figures for the recorded epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from google.protobuf.message import DecodeError
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


LOSS_KEYS = (
    "train_loss/total_loss",
    "train_loss/monitor_loss",
    "train_loss/success_loss",
    "train_loss/fail_loss",
    "train_loss/reg_loss",
    "train_loss/hard_neg_loss",
)


def _decode_history(run_path: Path) -> list[dict[str, Any]]:
    """Read all history rows from one W&B offline run."""

    rows: list[dict[str, Any]] = []
    store = DataStore()
    store.open_for_scan(str(run_path))
    try:
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            try:
                record.ParseFromString(payload)
            except DecodeError:
                # A damaged/non-record payload should not silently become a
                # fabricated loss value.  Continue scanning the next payload.
                continue
            if not record.history.item:
                continue
            row: dict[str, Any] = {}
            for item in record.history.item:
                key = item.key or "/".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    finally:
        store.close()
    return rows


def _loss_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select loss rows and assign the explicit training epoch."""

    losses = [row for row in history if "train_loss/total_loss" in row]
    epoch_rows = {
        int(row["epoch"]): row
        for row in history
        if isinstance(row.get("epoch"), (int, float))
    }
    output: list[dict[str, Any]] = []
    for epoch, row in enumerate(losses, start=1):
        epoch_meta = epoch_rows.get(epoch, {})
        output.append(
            {
                "epoch": epoch,
                "wandb_step": row.get("_step"),
                "learning_rate": epoch_meta.get("learning_rate"),
                **{key: row.get(key) for key in LOSS_KEYS},
            }
        )
    return output


def _write_csv(path: Path, model_rows: dict[str, list[dict[str, Any]]]) -> None:
    columns = ["model", "epoch", "wandb_step", "learning_rate", *LOSS_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, rows in model_rows.items():
            for row in rows:
                writer.writerow({"model": model, **row})


def _plot_model(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    epochs = [row["epoch"] for row in rows]
    colors = {
        "train_loss/total_loss": "#111827",
        "train_loss/monitor_loss": "#2563eb",
        "train_loss/success_loss": "#059669",
        "train_loss/fail_loss": "#dc2626",
        "train_loss/reg_loss": "#9333ea",
        "train_loss/hard_neg_loss": "#d97706",
    }
    labels = {
        "train_loss/total_loss": "total",
        "train_loss/monitor_loss": "monitor",
        "train_loss/success_loss": "success",
        "train_loss/fail_loss": "fail",
        "train_loss/reg_loss": "regularization",
        "train_loss/hard_neg_loss": "hard negative",
    }
    for key in LOSS_KEYS:
        values = [row[key] for row in rows]
        if all(value is None for value in values):
            continue
        ax.plot(
            epochs,
            values,
            label=labels[key],
            color=colors[key],
            linewidth=2.2 if key == "train_loss/total_loss" else 1.2,
            alpha=0.95 if key == "train_loss/total_loss" else 0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("recorded loss")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=False)


def _plot_detail(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
    detail = [row for row in rows if row["epoch"] >= 2]
    _plot_model(ax, detail, title)


def _write_figures(output_dir: Path, model_rows: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_index, model in enumerate(("MLP", "LSTM")):
        rows = model_rows[model]
        _plot_model(axes[row_index, 0], rows, f"SAFE-{model}: all recorded epochs")
        _plot_detail(axes[row_index, 1], rows, f"SAFE-{model}: epochs 2–{rows[-1]['epoch']} detail")
    fig.suptitle("SAFE training loss curves from offline W&B history", fontsize=15)
    for suffix, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(output_dir / f"safe_training_loss_curves.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-run", type=Path, required=True)
    parser.add_argument("--lstm-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = {
        "MLP": _loss_rows(_decode_history(args.mlp_run)),
        "LSTM": _loss_rows(_decode_history(args.lstm_run)),
    }
    if any(len(rows) == 0 for rows in model_rows.values()):
        raise RuntimeError("No train_loss/total_loss rows were found in one or more runs")

    _write_csv(args.output_dir / "safe_training_loss_history.csv", model_rows)
    _write_figures(args.output_dir, model_rows)
    metadata = {
        "source": {
            "MLP": str(args.mlp_run),
            "LSTM": str(args.lstm_run),
        },
        "epochs": {model: len(rows) for model, rows in model_rows.items()},
        "loss_keys": {
            model: [key for key in LOSS_KEYS if any(row[key] is not None for row in rows)]
            for model, rows in model_rows.items()
        },
        "final_values": {
            model: {key: rows[-1][key] for key in LOSS_KEYS if rows[-1][key] is not None}
            for model, rows in model_rows.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
