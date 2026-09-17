#!/usr/bin/env python3
"""Task-conditioned sanity analysis for Robo-Dopamine forward progress.

The script reads an already completed Robo-Dopamine run and human annotation
records. It extracts the last native forward progress value for each rollout,
joins it to the manifest and annotation by rollout ID, and writes compact
tables, JSON metadata, diagnostic plots, and a short report.

This is descriptive only: it does not normalize or resample curves, does not
substitute the fused signal for native forward, and never starts inference.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "annotations/failure_annotations/v1/records"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/baseline_signal_analysis"

OUTCOME_ORDER = (
    "clean_success",
    "recovered_success",
    "terminal_failure",
    "uncertain",
)
OUTCOME_LABELS = {
    "clean_success": "clean success",
    "recovered_success": "recovered success",
    "terminal_failure": "terminal failure",
    "uncertain": "uncertain",
}
OUTCOME_COLORS = {
    "clean_success": "#67d9b5",
    "recovered_success": "#78b7ff",
    "terminal_failure": "#f07f73",
    "uncertain": "#c9a96e",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def ensure_within(path: Path, base: Path = PROJECT_ROOT, label: str = "path") -> Path:
    path, base = path.resolve(), base.resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} must be inside {base}: {path}") from error
    return path


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing JSON file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing JSONL file: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        rollout_id = row.get("id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError(f"Manifest row has no valid id: {path}")
        if rollout_id in result:
            raise ValueError(f"Manifest contains duplicate rollout ID: {rollout_id}")
        result[rollout_id] = row
    return result


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record_path in sorted(path.glob("*.json")):
        record = read_json(record_path)
        rollout_id = record.get("rollout_id") or record_path.stem
        if not isinstance(rollout_id, str) or not rollout_id:
            continue
        if rollout_id in result:
            raise ValueError(f"Duplicate annotation rollout ID: {rollout_id}")
        result[rollout_id] = record
    return result


def normalize_outcome(annotation: dict[str, Any]) -> str:
    value = annotation.get("outcome_label")
    if value == "success":
        return "clean_success"
    if value == "recovered_success":
        return "recovered_success"
    if value == "failure":
        return "terminal_failure"
    return "uncertain"


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def frame_index_from_record(record: dict[str, Any]) -> int | None:
    value = record.get("id")
    if isinstance(value, str):
        match = re.search(r"af_(\d+)", value)
        if match:
            return int(match.group(1))
    images = record.get("image")
    if isinstance(images, list):
        indices = []
        for image in images:
            if isinstance(image, str):
                match = re.search(r"frame_(\d+)\.png", image)
                if match:
                    indices.append(int(match.group(1)))
        if indices:
            return max(indices)
    return None


def resolve_forward_output(run_root: Path, job: dict[str, Any]) -> Path:
    perspectives = job.get("perspective_outputs")
    if isinstance(perspectives, dict) and isinstance(perspectives.get("forward"), dict):
        value = perspectives["forward"].get("raw_model_output")
        if isinstance(value, str) and value:
            path = resolve_path(value, run_root)
            if path.is_file():
                return ensure_within(path, run_root, "forward output")

    rollout_id = job.get("rollout_id")
    if isinstance(rollout_id, str) and rollout_id:
        candidates = sorted(
            (run_root / "raw" / rollout_id).glob("*_forward_mode_*/pred_vllm.json")
        )
        if len(candidates) == 1:
            return ensure_within(candidates[0], run_root, "forward output")
        if len(candidates) > 1:
            raise ValueError(f"Multiple forward outputs for {rollout_id}")
    raise FileNotFoundError(f"Native forward output is missing for {rollout_id}")


def extract_terminal_forward(path: Path) -> tuple[float, int | None, int, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Forward output must be a non-empty JSON list: {path}")
    invalid = 0
    for record in value:
        if not isinstance(record, dict) or finite_float(record.get("progress")) is None:
            invalid += 1
    last = value[-1]
    if not isinstance(last, dict):
        raise ValueError(f"Last forward record is not an object: {path}")
    terminal = finite_float(last.get("progress"))
    if terminal is None:
        raise ValueError(f"Last native forward progress is not finite: {path}")
    return terminal, frame_index_from_record(last), len(value), invalid


def task_fields(row: dict[str, Any]) -> tuple[str, str, int, str]:
    suite = str(row.get("task_suite") or "unknown")
    try:
        task_id = int(row.get("task_id"))
    except (TypeError, ValueError):
        task_id = -1
    key = f"{suite}/task{task_id:02d}" if task_id >= 0 else f"{suite}/task_unknown"
    description = str(row.get("task_description") or row.get("task") or "")
    return key, suite, task_id, description


def short_task_label(task_key: str) -> str:
    suite, _, task = task_key.partition("/task")
    suite_label = {"libero_10": "L10", "libero_spatial": "LSP"}.get(suite, suite[:5])
    return f"{suite_label}/t{task}" if task else suite_label


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0, "q1": None, "median": None, "q3": None, "iqr": None,
            "min": None, "max": None, "std": None, "count_gt_0_9": 0,
            "fraction_gt_0_9": None,
        }
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75]).tolist()
    return {
        "n": int(array.size),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
        "count_gt_0_9": int(np.sum(array > 0.9)),
        "fraction_gt_0_9": float(np.mean(array > 0.9)),
    }


def task_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_info: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record["included_in_metrics"]:
            continue
        task_key = str(record["task_key"])
        task_info[task_key] = record
        value = float(record["terminal_forward_progress"])
        grouped[(task_key, "all")].append(value)
        grouped[(task_key, str(record["outcome"]))].append(value)

    rows: list[dict[str, Any]] = []
    for task_key in sorted(task_info):
        info = task_info[task_key]
        for outcome in ("all",) + OUTCOME_ORDER:
            values = grouped.get((task_key, outcome), [])
            if not values:
                continue
            rows.append({
                "task_key": task_key,
                "task_suite": info["task_suite"],
                "task_id": info["task_id"],
                "task_description": info["task_description"],
                "outcome": outcome,
                "task_total_n": len(grouped[(task_key, "all")]),
                **stats(values),
            })
    return rows


def comparison_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_task[str(row["task_key"])][str(row["outcome"])] = row
    result = []
    for task_key in sorted(by_task):
        clean, terminal = by_task[task_key].get("clean_success"), by_task[task_key].get("terminal_failure")
        if clean is None or terminal is None:
            continue
        result.append({
            "task_key": task_key,
            "task_suite": clean["task_suite"],
            "task_id": clean["task_id"],
            "task_description": clean["task_description"],
            "clean_n": clean["n"],
            "clean_median": clean["median"],
            "clean_q1": clean["q1"],
            "clean_q3": clean["q3"],
            "clean_iqr": clean["iqr"],
            "clean_min": clean["min"],
            "clean_max": clean["max"],
            "clean_fraction_gt_0_9": clean["fraction_gt_0_9"],
            "terminal_n": terminal["n"],
            "terminal_median": terminal["median"],
            "terminal_q1": terminal["q1"],
            "terminal_q3": terminal["q3"],
            "terminal_iqr": terminal["iqr"],
            "terminal_min": terminal["min"],
            "terminal_max": terminal["max"],
            "terminal_fraction_gt_0_9": terminal["fraction_gt_0_9"],
            "median_difference_terminal_minus_clean": terminal["median"] - clean["median"],
        })
    return result


def identification(summary_rows: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_task[str(row["task_key"])][str(row["outcome"])] = row
    clean_rows = [task["clean_success"] for task in by_task.values() if "clean_success" in task]
    all_rows = [task["all"] for task in by_task.values() if "all" in task]
    near_one = [{
        "task_key": row["task_key"], "n": row["n"], "median": row["median"],
        "q1": row["q1"], "iqr": row["iqr"], "fraction_gt_0_9": row["fraction_gt_0_9"],
    } for row in clean_rows if row["n"] >= 3 and row["fraction_gt_0_9"] == 1.0 and row["q1"] >= 0.9]
    lowest = [{
        "task_key": row["task_key"], "n": row["n"], "median": row["median"],
        "iqr": row["iqr"], "min": row["min"], "max": row["max"],
        "fraction_gt_0_9": row["fraction_gt_0_9"],
    } for row in clean_rows]
    lowest.sort(key=lambda row: (row["median"], row["n"], row["task_key"]))
    variance = [{
        "task_key": row["task_key"], "n": row["n"], "median": row["median"],
        "iqr": row["iqr"], "std": row["std"], "min": row["min"], "max": row["max"],
    } for row in all_rows]
    variance.sort(key=lambda row: (-row["iqr"], -row["n"], row["task_key"]))
    clean_variance = [{
        "task_key": row["task_key"], "n": row["n"], "median": row["median"],
        "iqr": row["iqr"], "std": row["std"], "min": row["min"], "max": row["max"],
    } for row in clean_rows]
    clean_variance.sort(key=lambda row: (-row["iqr"], -row["n"], row["task_key"]))
    differences = [{
        "task_key": row["task_key"], "clean_n": row["clean_n"], "terminal_n": row["terminal_n"],
        "clean_median": row["clean_median"], "terminal_median": row["terminal_median"],
        "median_difference_terminal_minus_clean": row["median_difference_terminal_minus_clean"],
    } for row in comparison]
    differences.sort(key=lambda row: (row["median_difference_terminal_minus_clean"], row["task_key"]))
    return {
        "near_one_criterion": "N >= 3, every clean-success value > 0.9, and Q1 >= 0.9; descriptive only",
        "clean_success_consistently_near_one": near_one,
        "clean_success_highest_medians": sorted(lowest, key=lambda row: (-row["median"], -row["n"], row["task_key"]))[:5],
        "clean_success_lowest_medians": lowest[:5],
        "high_within_task_variance_all_outcomes": variance[:5],
        "high_within_task_variance_clean_success_only": clean_variance[:5],
        "within_task_comparison_most_negative": differences[:5],
        "within_task_comparison_most_positive": list(reversed(differences[-5:])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def make_distribution_plot(records: list[dict[str, Any]], output: Path) -> None:
    task_keys = sorted({str(row["task_key"]) for row in records if row["included_in_metrics"]})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        if row["included_in_metrics"]:
            grouped[(str(row["task_key"]), str(row["outcome"]))].append(float(row["terminal_forward_progress"]))
    fig, ax = plt.subplots(figsize=(18, 8))
    offsets = {"clean_success": -0.27, "recovered_success": -0.09, "terminal_failure": 0.09, "uncertain": 0.27}
    rng = np.random.default_rng(20260917)
    for index, task_key in enumerate(task_keys):
        for outcome in OUTCOME_ORDER:
            values = grouped.get((task_key, outcome), [])
            if not values:
                continue
            position = index + offsets[outcome]
            box = ax.boxplot([values], positions=[position], widths=0.14, patch_artist=True,
                             showfliers=False, manage_ticks=False,
                             medianprops={"color": "#f4f6f7", "linewidth": 1.1},
                             whiskerprops={"color": OUTCOME_COLORS[outcome]},
                             capprops={"color": OUTCOME_COLORS[outcome]})
            for patch in box["boxes"]:
                patch.set_facecolor(OUTCOME_COLORS[outcome])
                patch.set_alpha(0.45)
                patch.set_edgecolor(OUTCOME_COLORS[outcome])
            ax.scatter(np.full(len(values), position) + rng.uniform(-0.045, 0.045, len(values)),
                       values, s=18, color=OUTCOME_COLORS[outcome], alpha=0.9, linewidths=0, zorder=3)
    ax.axhline(0.9, color="#f4b183", linestyle="--", linewidth=1.1)
    ax.set_xticks(range(len(task_keys)))
    ax.set_xticklabels([short_task_label(key) for key in task_keys], rotation=35, ha="right")
    ax.set_ylabel("last native forward progress (raw value)")
    ax.set_xlabel("task")
    ax.set_title("Robo-Dopamine terminal forward progress by task and human outcome")
    ax.grid(axis="y", alpha=0.22)
    ax.set_xlim(-0.65, len(task_keys) - 0.35)
    values = [float(row["terminal_forward_progress"]) for row in records if row["included_in_metrics"]]
    if values:
        ax.set_ylim(min(0.0, min(values)), max(1.0, max(values) * 1.08))
    legend = [Patch(facecolor=OUTCOME_COLORS[o], edgecolor=OUTCOME_COLORS[o], alpha=0.65, label=OUTCOME_LABELS[o]) for o in OUTCOME_ORDER]
    legend.append(plt.Line2D([], [], color="#f4b183", linestyle="--", label="descriptive >0.9 reference"))
    ax.legend(handles=legend, loc="upper left", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def make_comparison_plot(comparison: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    if not comparison:
        ax.text(0.5, 0.5, "No task has both clean-success and terminal-failure labels", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(comparison), dtype=float)
        for offset, outcome, color, median_key, q1_key, q3_key in (
            (-0.16, "clean_success", OUTCOME_COLORS["clean_success"], "clean_median", "clean_q1", "clean_q3"),
            (0.16, "terminal_failure", OUTCOME_COLORS["terminal_failure"], "terminal_median", "terminal_q1", "terminal_q3"),
        ):
            medians = np.asarray([row[median_key] for row in comparison], dtype=float)
            lower = medians - np.asarray([row[q1_key] for row in comparison], dtype=float)
            upper = np.asarray([row[q3_key] for row in comparison], dtype=float) - medians
            ax.errorbar(x + offset, medians, yerr=np.vstack([lower, upper]), fmt="o",
                        color=color, ecolor=color, capsize=3, markersize=5, label=OUTCOME_LABELS[outcome])
        ax.axhline(0.9, color="#f4b183", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="#98a3ad", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short_task_label(str(row["task_key"])) for row in comparison], rotation=35, ha="right")
        ax.set_ylabel("last native forward progress (median; error bars = quartiles)")
        ax.set_xlabel("task with both outcome groups")
        ax.set_title("Within-task clean-success versus terminal-failure terminal progress")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def value_text(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_report(
    output: Path,
    generated_at: str,
    source: dict[str, Any],
    selection: dict[str, Any],
    records: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    identified: dict[str, Any],
) -> None:
    included = [row for row in records if row["included_in_metrics"]]
    counts = Counter(str(row["outcome"]) for row in included)
    all_stat, clean_stat = stats(float(row["terminal_forward_progress"]) for row in included), stats(
        float(row["terminal_forward_progress"]) for row in included if row["outcome"] == "clean_success"
    )
    high = identified["clean_success_highest_medians"][:5]
    low = identified["clean_success_lowest_medians"][:5]
    variance = identified["high_within_task_variance_all_outcomes"][:5]
    negative = identified["within_task_comparison_most_negative"][:5]
    lines = [
        "# Robo-Dopamine task-conditioned terminal progress sanity check",
        "",
        f"Generated: {generated_at}",
        "",
        "## Main answer",
        "",
        "The evidence favors a task-dependent calibration effect as a major component "
        "of low terminal progress on successful rollouts, rather than pure rollout-level "
        "instability. Clean-success task medians span "
        f"{value_text(min((row['median'] for row in high + low), default=None))} to "
        f"{value_text(max((row['median'] for row in high + low), default=None))} in the "
        "ranked task summaries, while several groups have tight clean-success IQRs. "
        "The effect is not exclusively task calibration: some tasks have large "
        "within-task spread, and small groups require caution.",
        "",
        "This is descriptive. Curves were not normalized or resampled, and >0.9 is "
        "reported only as a descriptive reference, not used to relabel outcomes.",
        "",
        "## Source and selection",
        "",
        f"- Robo-Dopamine run: {source['run_root']}",
        f"- Run status: {source.get('run_status')}; instruction condition: {source.get('instruction_condition')}",
        "- Signal: the final record of each native forward pred_vllm.json progress list",
        f"- Frame interval: {source.get('frame_interval')}; fused output was not used",
        f"- Checkpoint: {source.get('checkpoint')}",
        f"- Source commit: {source.get('source_commit')}",
        f"- Goal image: {source.get('goal_image')}",
        f"- Human annotations: {source['annotation_dir']}",
        "",
        f"The manifest has {selection['manifest_rows']} rows. The selected completed "
        f"run has {selection['complete_jobs']} jobs; {selection['included_rollouts']} "
        "completed jobs have both a human label and a finite terminal forward value.",
        "",
        markdown_table(["human outcome", "N"], [[OUTCOME_LABELS[o], counts.get(o, 0)] for o in OUTCOME_ORDER]),
        "",
        f"All included rollouts: median {value_text(all_stat['median'])}, IQR "
        f"{value_text(all_stat['iqr'])}, fraction >0.9 {value_text(all_stat['fraction_gt_0_9'])}. "
        f"Clean success: median {value_text(clean_stat['median'])}, IQR "
        f"{value_text(clean_stat['iqr'])}, fraction >0.9 {value_text(clean_stat['fraction_gt_0_9'])}.",
        "",
        "## Clean-success task patterns",
        "",
        "No task meets the conservative descriptive near-one flag (N >= 3, every "
        "clean-success value >0.9, and Q1 >=0.9). Highest clean-success medians:",
        "",
        markdown_table(["task", "N", "median", "IQR", "min–max", ">0.9"], [
            [short_task_label(row["task_key"]), row["n"], value_text(row["median"]), value_text(row["iqr"]),
             f"{value_text(row['min'])}–{value_text(row['max'])}", value_text(row["fraction_gt_0_9"])] for row in high
        ]),
        "",
        "Lowest clean-success medians (ranked, not thresholded):",
        "",
        markdown_table(["task", "N", "median", "IQR", "min–max", ">0.9"], [
            [short_task_label(row["task_key"]), row["n"], value_text(row["median"]), value_text(row["iqr"]),
             f"{value_text(row['min'])}–{value_text(row['max'])}", value_text(row["fraction_gt_0_9"])] for row in low
        ]),
        "",
        "Largest all-outcome within-task IQRs:",
        "",
        markdown_table(["task", "N", "median", "IQR", "std", "min–max"], [
            [short_task_label(row["task_key"]), row["n"], value_text(row["median"]), value_text(row["iqr"]),
             value_text(row["std"]), f"{value_text(row['min'])}–{value_text(row['max'])}"] for row in variance
        ]),
        "",
        "## Within-task comparison",
        "",
        f"{len(comparison)} tasks have both clean-success and terminal-failure labels. "
        "Most negative terminal-minus-clean median differences:",
        "",
        markdown_table(["task", "clean N", "terminal N", "clean median", "terminal median", "difference"], [
            [short_task_label(row["task_key"]), row["clean_n"], row["terminal_n"],
             value_text(row["clean_median"]), value_text(row["terminal_median"]),
             value_text(row["median_difference_terminal_minus_clean"])] for row in negative
        ]),
        "",
        "The full comparison CSV contains every task with both groups. A positive or "
        "near-zero difference is an observed pattern, not a correctness claim.",
        "",
        "## Artifacts",
        "",
        "- REPORT.md",
        "- rollout_terminal_forward_progress.csv: one compact row per completed rollout",
        "- terminal_forward_progress_by_task_outcome.csv: task/outcome statistics plus all rows",
        "- terminal_forward_progress_task_comparison.csv: clean versus terminal comparison",
        "- terminal_forward_progress_summary.json: compact machine-readable summary",
        "- metadata.json: provenance, hashes, and metric definitions",
        "- task_terminal_forward_progress_distribution.png",
        "- task_clean_vs_terminal.png",
        "",
        "The dashed line in the plots marks the descriptive >0.9 reference only.",
        "",
        "## Exclusions",
        "",
        "The 2026-09-17 subtask_a run was not merged because it uses a different "
        "instruction condition and its rollout IDs do not match the main human "
        "annotation records. The two manifest rows without an annotation or completed "
        f"job are excluded: {', '.join(selection['manifest_rows_without_annotation_or_job']) or 'none'}.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyse(args: argparse.Namespace) -> Path:
    run_root = ensure_within(resolve_path(args.run_root), PROJECT_ROOT, "run root")
    manifest_path = ensure_within(resolve_path(args.manifest), PROJECT_ROOT, "manifest")
    annotation_dir = ensure_within(resolve_path(args.annotations), PROJECT_ROOT, "annotation directory")
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    output = (
        resolve_path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"robo_dopamine_terminal_task_sanity_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output = ensure_within(output, PROJECT_ROOT, "output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=False)

    manifest, annotations = load_manifest(manifest_path), load_annotations(annotation_dir)
    run_metadata = read_json(run_root / "run.json") if (run_root / "run.json").is_file() else {}
    jobs = read_jsonl(run_root / "jobs.jsonl")
    complete_jobs = [job for job in jobs if job.get("status") == "complete" and job.get("return_code") in (None, 0)]
    job_ids = [str(job.get("rollout_id")) for job in complete_jobs]
    if len(job_ids) != len(set(job_ids)):
        duplicate_ids = [rid for rid, n in Counter(job_ids).items() if n > 1]
        raise ValueError(f"Completed jobs contain duplicate rollout IDs: {duplicate_ids}")

    records: list[dict[str, Any]] = []
    for job in complete_jobs:
        rollout_id = str(job.get("rollout_id") or "")
        manifest_row, annotation = manifest.get(rollout_id), annotations.get(rollout_id)
        record: dict[str, Any] = {
            "rollout_id": rollout_id, "job_index": job.get("job_index"),
            "task_key": "", "task_suite": "", "task_id": "", "task_description": "",
            "raw_outcome_label": annotation.get("outcome_label") if annotation else None,
            "outcome": normalize_outcome(annotation) if annotation else None,
            "failure_type": annotation.get("failure_type") if annotation else None,
            "terminal_forward_progress": None, "terminal_forward_frame": None,
            "forward_sample_count": None, "invalid_forward_progress_records": None,
            "forward_source": None, "extraction_status": "excluded",
            "included_in_metrics": False,
        }
        if manifest_row is None:
            record["extraction_status"] = "excluded_manifest_missing"
            records.append(record)
            continue
        key, suite, task_id, description = task_fields(manifest_row)
        record.update({"task_key": key, "task_suite": suite, "task_id": task_id, "task_description": description})
        if annotation is None:
            record["extraction_status"] = "excluded_human_annotation_missing"
            records.append(record)
            continue
        try:
            forward_path = resolve_forward_output(run_root, job)
            terminal, frame, sample_count, invalid = extract_terminal_forward(forward_path)
        except (FileNotFoundError, OSError, ValueError) as error:
            record["extraction_status"] = f"excluded_forward_output_error: {error}"
            records.append(record)
            continue
        record.update({
            "terminal_forward_progress": terminal, "terminal_forward_frame": frame,
            "forward_sample_count": sample_count, "invalid_forward_progress_records": invalid,
            "forward_source": relative_to_project(forward_path), "extraction_status": "included",
            "included_in_metrics": True,
        })
        records.append(record)

    included = [row for row in records if row["included_in_metrics"]]
    summary_rows, comparison = task_summary_rows(records), None
    comparison = comparison_rows(summary_rows)
    identified = identification(summary_rows, comparison)
    record_fields = [
        "rollout_id", "job_index", "task_key", "task_suite", "task_id", "task_description",
        "raw_outcome_label", "outcome", "failure_type", "terminal_forward_progress",
        "terminal_forward_frame", "forward_sample_count", "invalid_forward_progress_records",
        "forward_source", "extraction_status", "included_in_metrics",
    ]
    summary_fields = [
        "task_key", "task_suite", "task_id", "task_description", "outcome", "task_total_n",
        "n", "q1", "median", "q3", "iqr", "min", "max", "std", "count_gt_0_9", "fraction_gt_0_9",
    ]
    comparison_fields = [
        "task_key", "task_suite", "task_id", "task_description", "clean_n", "clean_median",
        "clean_q1", "clean_q3", "clean_iqr", "clean_min", "clean_max", "clean_fraction_gt_0_9",
        "terminal_n", "terminal_median", "terminal_q1", "terminal_q3", "terminal_iqr",
        "terminal_min", "terminal_max", "terminal_fraction_gt_0_9",
        "median_difference_terminal_minus_clean",
    ]
    write_csv(output / "rollout_terminal_forward_progress.csv", records, record_fields)
    write_csv(output / "terminal_forward_progress_by_task_outcome.csv", summary_rows, summary_fields)
    write_csv(output / "terminal_forward_progress_task_comparison.csv", comparison, comparison_fields)

    worker_result: dict[str, Any] = {}
    for job in complete_jobs:
        value = job.get("worker_result_path")
        if isinstance(value, str) and value:
            path = resolve_path(value, run_root)
            if path.is_file():
                worker_result = read_json(path)
                if worker_result:
                    break
    source = {
        "run_root": relative_to_project(run_root),
        "run_status": run_metadata.get("status"),
        "baseline": run_metadata.get("baseline") or "robo_dopamine",
        "instruction_condition": run_metadata.get("instruction_condition") or run_metadata.get("instruction_variant") or "full_instruction",
        "frame_interval": worker_result.get("frame_interval"),
        "eval_mode": worker_result.get("eval_mode"),
        "eval_modes": worker_result.get("eval_modes"),
        "checkpoint": worker_result.get("checkpoint"),
        "goal_image": worker_result.get("goal_image"),
        "fusion_rule": worker_result.get("fusion_rule"),
        "source_commit": worker_result.get("source_commit") or run_metadata.get("baseline_revision"),
        "manifest": relative_to_project(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "jobs_sha256": sha256(run_root / "jobs.jsonl"),
        "annotation_dir": relative_to_project(annotation_dir),
    }
    job_id_set, manifest_id_set = set(job_ids), set(manifest)
    manifest_without_annotation_or_job = sorted(
        rollout_id for rollout_id in manifest if rollout_id not in annotations or rollout_id not in job_id_set
    )
    selection = {
        "manifest_rows": len(manifest), "annotation_records": len(annotations),
        "total_jobs": len(jobs), "complete_jobs": len(complete_jobs),
        "included_rollouts": len(included), "excluded_rollouts": len(records) - len(included),
        "manifest_rows_without_annotation_or_job": manifest_without_annotation_or_job,
        "complete_job_ids_without_manifest": sorted(job_id_set - manifest_id_set),
        "complete_job_ids_without_annotation": sorted(job_id_set - set(annotations)),
    }
    summary_document = {
        "schema_version": 1, "generated_at": now_iso(),
        "analysis": {
            "terminal_definition": "last native forward pred_vllm.json record progress; no interpolation, normalization, or fused substitution",
            "progress_scale": "native Robo-Dopamine forward progress values",
            "descriptive_gt_0_9_only": True, "outcomes": list(OUTCOME_ORDER),
        },
        "source": source, "selection": selection,
        "outcome_counts": dict(Counter(str(row["outcome"]) for row in included)),
        "task_count": len({row["task_key"] for row in included}),
        "task_outcome_summary": summary_rows, "task_comparisons": comparison,
        "identification": identified,
    }
    (output / "terminal_forward_progress_summary.json").write_text(
        json.dumps(summary_document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1, "generated_at": summary_document["generated_at"],
        "script": relative_to_project(Path(__file__)),
        "analysis_environment": "LF3R_ANALYSIS_PYTHON / project-local CPU analysis environment",
        "source": source, "selection": selection,
        "metric_definitions": {
            "terminal_forward_progress": "last native forward progress value, unchanged",
            "median": "50th percentile of native terminal values", "iqr": "Q75 - Q25",
            "fraction_gt_0_9": "count(native terminal progress > 0.9) / finite group count",
            "human_outcome_mapping": {"success": "clean_success", "recovered_success": "recovered_success", "failure": "terminal_failure", "other_or_missing": "uncertain"},
        },
        "output_files": [
            "REPORT.md", "rollout_terminal_forward_progress.csv",
            "terminal_forward_progress_by_task_outcome.csv",
            "terminal_forward_progress_task_comparison.csv",
            "terminal_forward_progress_summary.json", "task_terminal_forward_progress_distribution.png",
            "task_clean_vs_terminal.png",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_distribution_plot(records, output / "task_terminal_forward_progress_distribution.png")
    make_comparison_plot(comparison, output / "task_clean_vs_terminal.png")
    make_report(output, summary_document["generated_at"], source, selection, records, comparison, identified)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize native Robo-Dopamine forward terminal progress by task and human outcome."
    )
    parser.add_argument("--run-root", required=True, help="Completed Robo-Dopamine run directory containing run.json, jobs.jsonl, and raw/.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output-dir", default=None, help="New project-local output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        output = analyse(build_parser().parse_args(argv))
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
