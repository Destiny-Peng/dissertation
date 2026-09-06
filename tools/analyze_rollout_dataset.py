#!/usr/bin/env python3
"""Generate descriptive statistics for the annotated LF3R natural rollouts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/manifest.jsonl"
ANNOTATION_DIR = PROJECT_ROOT / "annotations/failure_annotations/v1/records"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
JSON_PATH = OUTPUT_DIR / "lf3r_rollout_dataset_summary.json"
CSV_PATH = OUTPUT_DIR / "lf3r_rollout_task_statistics.csv"
CASES_PATH = OUTPUT_DIR / "lf3r_rollout_representative_cases.jsonl"
REPORT_PATH = OUTPUT_DIR / "lf3r_rollout_dataset_summary.md"

OUTCOME_MAP = {
    "success": "clean_success",
    "recovered_success": "recovered_success",
    "failure": "terminal_failure",
    "uncertain": "uncertain",
}
OUTCOME_ORDER = ("clean_success", "recovered_success", "terminal_failure", "uncertain")
TIMING_FIELDS = (
    "causal_onset_frame",
    "observable_onset_frame",
    "terminal_failure_frame",
    "recovery_frame",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    return {
        "count": len(values),
        "min": rounded(min(values)),
        "p25": rounded(percentile(values, 0.25)),
        "median": rounded(median(values)),
        "mean": rounded(mean(values)),
        "p75": rounded(percentile(values, 0.75)),
        "max": rounded(max(values)),
    }


def distribution(counter: Counter, total: int, keys: tuple[str, ...] | None = None) -> dict:
    ordered_keys = keys or tuple(sorted(counter))
    return {
        key: {"count": counter.get(key, 0), "percent": rounded(100 * counter.get(key, 0) / total, 2) if total else None}
        for key in ordered_keys
    }


def effective_events(annotation: dict) -> list[dict]:
    events = annotation.get("failure_events") or []
    if events:
        return [{**event, "source": "failure_events"} for event in events]
    if any(annotation.get(field) is not None for field in TIMING_FIELDS):
        return [{
            "failure_type": annotation.get("failure_type"),
            **{field: annotation.get(field) for field in TIMING_FIELDS},
            "notes": annotation.get("notes", ""),
            "source": "legacy_top_level",
        }]
    return []


def event_text(event: dict, index: int) -> str:
    points = []
    for field, label in (
        ("causal_onset_frame", "causal"),
        ("observable_onset_frame", "observable"),
        ("terminal_failure_frame", "terminal"),
        ("recovery_frame", "recovery"),
    ):
        if event.get(field) is not None:
            points.append(f"{label} f{event[field]}")
    timing = ", ".join(points) if points else "no timing points"
    return f"event {index} {event.get('failure_type', 'other')}: {timing}"


def make_case(annotation: dict, record: dict, reason: str) -> dict:
    events = effective_events(annotation)
    return {
        "rollout_id": record["id"],
        "selection_reason": reason,
        "task_suite": record["task_suite"],
        "task_id": record["task_id"],
        "episode_index": record["episode_index"],
        "task_description": record["task_description"],
        "outcome": OUTCOME_MAP[annotation["outcome_label"]],
        "evaluator_outcome": record["ground_truth_outcome"],
        "primary_failure_type": annotation["failure_type"],
        "confidence": annotation.get("confidence"),
        "total_frames": record["total_frames"],
        "fps": record["fps"],
        "video_path": record["video_path"],
        "events": [{key: value for key, value in event.items() if key != "source"} for event in events],
        "annotation_notes": annotation.get("notes", ""),
        "summary": "; ".join(event_text(event, index) for index, event in enumerate(events, 1)),
    }


def fmt_percent(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.1f}%)" if total else f"{count}"


def fmt_metric(metric: dict) -> str:
    if not metric["count"]:
        return "—"
    return f"{metric['median']:.2f} [{metric['p25']:.2f}, {metric['p75']:.2f}]"


def main() -> None:
    manifest = load_jsonl(MANIFEST_PATH)
    manifest_by_id = {record["id"]: record for record in manifest}
    if len(manifest_by_id) != len(manifest):
        raise RuntimeError("Manifest contains duplicate rollout IDs")

    annotations = {}
    for path in sorted(ANNOTATION_DIR.glob("*.json")):
        annotation = json.loads(path.read_text())
        annotations[annotation["rollout_id"]] = annotation
    orphan_ids = sorted(set(annotations) - set(manifest_by_id))
    if orphan_ids:
        raise RuntimeError(f"Annotations missing from manifest: {orphan_ids}")

    cohort = [record for record in manifest if record.get("analysis_partition") == "natural_observation"]
    excluded = [record for record in manifest if record.get("analysis_partition") != "natural_observation"]
    if len(cohort) != 136:
        raise RuntimeError(f"Expected 136 natural rollouts, found {len(cohort)}")
    missing = [record["id"] for record in cohort if record["id"] not in annotations]
    if missing:
        raise RuntimeError(f"Natural rollouts missing annotations: {missing}")
    cohort_annotations = {record["id"]: annotations[record["id"]] for record in cohort}
    incomplete = [rollout_id for rollout_id, value in cohort_annotations.items() if value.get("review_status") != "complete"]
    if incomplete:
        raise RuntimeError(f"Natural annotations are not complete: {incomplete}")

    canonical_outcomes = Counter(OUTCOME_MAP[value["outcome_label"]] for value in cohort_annotations.values())
    resolved_total = len(cohort) - canonical_outcomes["uncertain"]
    evaluator_outcomes = Counter(record["ground_truth_outcome"] for record in cohort)
    cross_tab = defaultdict(Counter)
    for record in cohort:
        cross_tab[OUTCOME_MAP[cohort_annotations[record["id"]]["outcome_label"]]][record["ground_truth_outcome"]] += 1

    event_rows = []
    legacy_event_count = 0
    for record in cohort:
        annotation = cohort_annotations[record["id"]]
        for event_index, event in enumerate(effective_events(annotation), 1):
            if event["source"] == "legacy_top_level":
                legacy_event_count += 1
            for field in TIMING_FIELDS:
                frame = event.get(field)
                if frame is not None and not 0 <= frame < record["total_frames"]:
                    raise RuntimeError(f"Out-of-range {field} in {record['id']}: {frame}")
            event_rows.append({"record": record, "annotation": annotation, "event": event, "event_index": event_index})

    def timing_metric(field: str) -> dict:
        rows = [row for row in event_rows if row["event"].get(field) is not None]
        frames = [float(row["event"][field]) for row in rows]
        seconds = [row["event"][field] / float(row["record"]["fps"]) for row in rows]
        fractions = [row["event"][field] / max(1, row["record"]["total_frames"] - 1) for row in rows]
        return {
            "coverage": {"annotated_events": len(rows), "all_effective_events": len(event_rows), "percent": rounded(100 * len(rows) / len(event_rows), 2)},
            "frames": describe(frames),
            "seconds": describe(seconds),
            "trajectory_fraction": describe(fractions),
        }

    def interval_metric(start: str, end: str) -> dict:
        rows = [row for row in event_rows if row["event"].get(start) is not None and row["event"].get(end) is not None]
        frame_values = [float(row["event"][end] - row["event"][start]) for row in rows]
        second_values = [(row["event"][end] - row["event"][start]) / float(row["record"]["fps"]) for row in rows]
        return {"paired_events": len(rows), "frames": describe(frame_values), "seconds": describe(second_values)}

    timing = {field: timing_metric(field) for field in TIMING_FIELDS}
    timing["intervals"] = {
        "causal_to_observable": interval_metric("causal_onset_frame", "observable_onset_frame"),
        "causal_to_terminal": interval_metric("causal_onset_frame", "terminal_failure_frame"),
        "causal_to_recovery": interval_metric("causal_onset_frame", "recovery_frame"),
        "observable_to_recovery": interval_metric("observable_onset_frame", "recovery_frame"),
    }
    terminal_rollout_ids = {row["record"]["id"] for row in event_rows if row["event"].get("terminal_failure_frame") is not None}
    timing["terminal_outcome_coverage"] = {
        "terminal_failure_rollouts": canonical_outcomes["terminal_failure"],
        "rollouts_with_terminal_frame": sum(
            rollout_id in terminal_rollout_ids
            for rollout_id, annotation in cohort_annotations.items()
            if OUTCOME_MAP[annotation["outcome_label"]] == "terminal_failure"
        ),
    }

    rollout_failure_types = Counter(annotation["failure_type"] for annotation in cohort_annotations.values())
    non_clean_failure_types = Counter(
        annotation["failure_type"]
        for annotation in cohort_annotations.values()
        if OUTCOME_MAP[annotation["outcome_label"]] != "clean_success"
    )
    event_failure_types = Counter(row["event"].get("failure_type", "other") for row in event_rows)

    grouped = defaultdict(list)
    for record in cohort:
        grouped[(record["task_suite"], record["task_id"], record["task_description"])].append(record)
    task_rows = []
    for (suite, task_id, description), records in sorted(grouped.items()):
        counts = Counter(OUTCOME_MAP[cohort_annotations[record["id"]]["outcome_label"]] for record in records)
        resolved = len(records) - counts["uncertain"]
        task_event_rows = [row for row in event_rows if row["record"]["id"] in {record["id"] for record in records}]
        causal_fractions = [
            row["event"]["causal_onset_frame"] / max(1, row["record"]["total_frames"] - 1)
            for row in task_event_rows if row["event"].get("causal_onset_frame") is not None
        ]
        task_rows.append({
            "task_suite": suite,
            "task_id": task_id,
            "task_description": description,
            "total": len(records),
            **{outcome: counts[outcome] for outcome in OUTCOME_ORDER},
            "resolved_success_rate_percent": rounded(100 * (counts["clean_success"] + counts["recovered_success"]) / resolved, 2) if resolved else None,
            "evaluator_success": sum(record["ground_truth_outcome"] == "success" for record in records),
            "evaluator_failure": sum(record["ground_truth_outcome"] == "failure" for record in records),
            "effective_failure_events": len(task_event_rows),
            "median_causal_onset_fraction": rounded(median(causal_fractions)) if causal_fractions else None,
        })
    if sum(row["total"] for row in task_rows) != len(cohort):
        raise RuntimeError("Task totals do not sum to cohort size")

    recovered_cases = [
        make_case(cohort_annotations[record["id"]], record, "all manually identified recovered-success cases")
        for record in cohort if OUTCOME_MAP[cohort_annotations[record["id"]]["outcome_label"]] == "recovered_success"
    ]
    failure_case_ids = [
        ("libero_10-task08-ep006-natural-1f3811f27d", "largest annotated failed-attempt sequence (four events)"),
        ("libero_10-task03-ep007-natural-2dd9870cd4", "mixed dropped-object, grasp, and control-error sequence"),
        ("libero_10-task00-ep001-natural-477e2f3af1", "terminal frame and free-text cause are both annotated"),
        ("libero_10-task05-ep003-natural-fefb27649d", "high-confidence placement-failure example"),
    ]
    representative_failures = [
        make_case(cohort_annotations[rollout_id], manifest_by_id[rollout_id], reason)
        for rollout_id, reason in failure_case_ids if rollout_id in cohort_annotations
    ]
    representative_cases = recovered_cases + representative_failures

    summary = {
        "analysis_schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "annotations": str(ANNOTATION_DIR.relative_to(PROJECT_ROOT)),
        },
        "cohort": {
            "manifest_rollouts": len(manifest),
            "included_natural_rollouts": len(cohort),
            "excluded_controlled_rollouts": len(excluded),
            "complete_annotations": len(cohort_annotations),
            "task_suites": dict(sorted(Counter(record["task_suite"] for record in cohort).items())),
            "dataset_roles": dict(sorted(Counter(record["dataset_role"] for record in cohort).items())),
            "outcome_normalization": OUTCOME_MAP,
        },
        "trajectory_outcomes": {
            "all_136": distribution(canonical_outcomes, len(cohort), OUTCOME_ORDER),
            "resolved_only": distribution(canonical_outcomes, resolved_total, OUTCOME_ORDER[:-1]),
            "resolved_denominator": resolved_total,
            "evaluator_filename_outcomes": distribution(evaluator_outcomes, len(cohort)),
            "annotation_by_evaluator_outcome": {key: dict(sorted(value.items())) for key, value in sorted(cross_tab.items())},
        },
        "failure_types": {
            "rollout_level_all": distribution(rollout_failure_types, len(cohort)),
            "rollout_level_non_clean": distribution(non_clean_failure_types, len(cohort) - canonical_outcomes["clean_success"]),
            "event_level": distribution(event_failure_types, len(event_rows)),
        },
        "failure_events": {
            "effective_event_count": len(event_rows),
            "rollouts_with_effective_events": len({row["record"]["id"] for row in event_rows}),
            "v2_array_events": len(event_rows) - legacy_event_count,
            "legacy_top_level_events": legacy_event_count,
            "events_per_rollout": dict(sorted(Counter(len(effective_events(value)) for value in cohort_annotations.values()).items())),
        },
        "timing_statistics": timing,
        "task_statistics": task_rows,
        "representative_cases": representative_cases,
        "limitations": [
            "Thirteen complete annotations retain the explicit uncertain outcome and are not forced into the three resolved outcome classes.",
            "Only two terminal-failure rollouts contain a terminal_failure_frame; terminal timing estimates therefore have very low coverage.",
            "Representative cases summarize annotation metadata and do not claim a new video-content review.",
            "Controlled injected rollouts are excluded from all natural-rollout rates.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    with CASES_PATH.open("w") as handle:
        for case in representative_cases:
            handle.write(json.dumps(case, sort_keys=False) + "\n")

    event_type_rows = sorted(event_failure_types.items(), key=lambda item: (-item[1], item[0]))
    report = [
        "# LF3R Rollout Dataset Summary",
        "",
        f"Generated from the versioned manifest and complete human annotations. The analysis covers **{len(cohort)} natural OpenVLA + LIBERO rollouts**: "
        f"{Counter(record['task_suite'] for record in cohort)['libero_10']} LIBERO-10 primary rollouts and "
        f"{Counter(record['task_suite'] for record in cohort)['libero_spatial']} LIBERO-Spatial reference rollouts. "
        f"The four controlled-injection rollouts in the 140-row manifest are excluded.",
        "",
        "## Outcome distribution",
        "",
        "Legacy `success` and `failure` labels are reported as `clean_success` and `terminal_failure`. "
        "The 13 explicit uncertain labels remain unresolved rather than being inferred from evaluator filenames.",
        "",
        "| Human outcome | Count (% of 136) | % of 123 resolved |",
        "| --- | ---: | ---: |",
    ]
    for outcome in OUTCOME_ORDER:
        count = canonical_outcomes[outcome]
        resolved_value = "—" if outcome == "uncertain" else f"{100 * count / resolved_total:.1f}%"
        report.append(f"| `{outcome}` | {fmt_percent(count, len(cohort))} | {resolved_value} |")
    report += [
        "",
        f"Evaluator filenames encode {evaluator_outcomes['success']} successes and {evaluator_outcomes['failure']} failures. "
        "These are retained separately from human labels; notably, recovered success is a human trajectory interpretation, not an evaluator-native class.",
        "",
        "## Failure types and event multiplicity",
        "",
        f"There are **{len(event_rows)} effective failure events across {len({row['record']['id'] for row in event_rows})} rollouts**. "
        f"This combines {len(event_rows) - legacy_event_count} v2 array events with {legacy_event_count} legacy top-level events. "
        f"Event counts per rollout are: " + ", ".join(f"{key} events: {value}" for key, value in sorted(Counter(len(effective_events(a)) for a in cohort_annotations.values()).items())) + ".",
        "",
        "| Event failure type | Events | Share |",
        "| --- | ---: | ---: |",
    ]
    for failure_type, count in event_type_rows:
        report.append(f"| `{failure_type}` | {count} | {100 * count / len(event_rows):.1f}% |")
    report += [
        "",
        "The rollout-level primary labels are: " + ", ".join(f"`{key}` {value}" for key, value in sorted(rollout_failure_types.items(), key=lambda item: (-item[1], item[0]))) + ". "
        "Rollout-level labels describe the primary diagnosis; event-level labels capture repeated or mixed failure episodes.",
        "",
        "## Timing statistics",
        "",
        "Medians below include IQR in brackets. Frame statistics are event-level; normalized timing divides by `total_frames - 1`, enabling comparison across trajectory lengths.",
        "",
        "| Timing field | Coverage (of 68 events) | Frame median [IQR] | Seconds median [IQR] | Trajectory fraction median [IQR] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in TIMING_FIELDS:
        metric = timing[field]
        report.append(
            f"| `{field}` | {metric['coverage']['annotated_events']} ({metric['coverage']['percent']:.1f}%) | "
            f"{fmt_metric(metric['frames'])} | {fmt_metric(metric['seconds'])} | {fmt_metric(metric['trajectory_fraction'])} |"
        )
    report += [
        "",
        f"Causal-to-observable lag is {fmt_metric(timing['intervals']['causal_to_observable']['frames'])} frames "
        f"({fmt_metric(timing['intervals']['causal_to_observable']['seconds'])} s; n={timing['intervals']['causal_to_observable']['paired_events']}). "
        f"Causal-to-recovery time is {fmt_metric(timing['intervals']['causal_to_recovery']['frames'])} frames "
        f"({fmt_metric(timing['intervals']['causal_to_recovery']['seconds'])} s; n={timing['intervals']['causal_to_recovery']['paired_events']}).",
        "",
        f"**Coverage warning:** only {timing['terminal_outcome_coverage']['rollouts_with_terminal_frame']} of "
        f"{timing['terminal_outcome_coverage']['terminal_failure_rollouts']} terminal-failure rollouts have an explicit terminal frame. "
        "The terminal timing row is descriptive of those two cases, not a dataset-wide estimate.",
        "",
        "## Task-wise statistics",
        "",
        "Resolved success rate counts clean plus recovered success and excludes uncertain outcomes from the denominator.",
        "",
        "| Suite | Task | N | Clean | Recovered | Terminal | Uncertain | Resolved success | Events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in task_rows:
        rate = "—" if row["resolved_success_rate_percent"] is None else f"{row['resolved_success_rate_percent']:.1f}%"
        report.append(
            f"| `{row['task_suite']}` | {row['task_id']} | {row['total']} | {row['clean_success']} | "
            f"{row['recovered_success']} | {row['terminal_failure']} | {row['uncertain']} | {rate} | {row['effective_failure_events']} |"
        )
    report += [
        "",
        "LIBERO-10 task 8 has the highest terminal-failure count (8/10); task 5 has the largest uncertain share (7/10). "
        "All 11 LIBERO-Spatial reference rollouts are annotated clean successes, so suite composition should be preserved when comparing rates.",
        "",
        "## Representative cases",
        "",
        "These examples are selected from annotation metadata; frame descriptions are not a new visual re-review.",
        "",
    ]
    for case in representative_cases:
        report.append(
            f"- `{case['rollout_id']}` — **{case['outcome']}**, {case['selection_reason']}. "
            f"{case['summary'] or 'No event timing annotated.'}"
        )
    report += [
        "",
        "## Interpretation limits",
        "",
        "- Thirteen annotations are explicitly uncertain; both all-rollout and resolved-only denominators are reported.",
        "- Terminal-frame coverage is too sparse for a robust terminal timing conclusion.",
        "- Four controlled interventions are excluded, and LIBERO-Spatial reference data should not be treated as additional LIBERO-10 trials.",
        "- This report analyzes existing metadata and annotations only; no rollout was regenerated or modified.",
        "",
        "Machine-readable details are in `lf3r_rollout_dataset_summary.json`, task rows in `lf3r_rollout_task_statistics.csv`, and case records in `lf3r_rollout_representative_cases.jsonl`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report))

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {CASES_PATH}")


if __name__ == "__main__":
    main()
