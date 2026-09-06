#!/usr/bin/env python3
"""Refine onset threshold and signed-direction statistics from existing metrics.

This tool deliberately reads only the existing event_metrics.jsonl and
clean_background_metrics.jsonl files (and validates the derived background
summary). It does not load baseline models or rerun inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "outputs/baseline_signal_analysis/full_136_20260827"
METHODS = ("safe", "procvlm", "rynnvalue", "robo_dopamine")
SIGNALS_BY_METHOD = {
    "safe": ("max_token_prob", "avg_token_prob", "max_token_entropy", "avg_token_entropy"),
    "procvlm": ("progress",),
    "rynnvalue": ("value",),
    "robo_dopamine": ("progress", "hop"),
}
OUTCOMES = ("recovered_success", "terminal_failure", "uncertain")
EXPECTED_FAILURE_DIRECTION = {
    "safe": "positive",
    "procvlm": "negative",
    "rynnvalue": "positive",
    "robo_dopamine": "negative",
}
SUMMARY_FIELDS = [
    "method",
    "signal",
    "outcome_group",
    "n_onset_events",
    "n_valid_response_magnitude_events",
    "clean_background_q95",
    "n_exceeding_q95",
    "exceeding_q95_fraction",
    "n_valid_signed_change_events",
    "signed_positive_count",
    "signed_negative_count",
    "signed_zero_count",
    "signed_positive_count_above_q95",
    "signed_negative_count_above_q95",
    "signed_zero_count_above_q95",
    "signed_positive_fraction_above_q95",
    "signed_negative_fraction_above_q95",
    "signed_zero_fraction_above_q95",
    "expected_failure_direction",
    "direction_consistency_count_all",
    "direction_consistency_fraction_all",
    "direction_consistency_count_above_q95",
    "direction_consistency_fraction_above_q95",
]
SECTION_BEGIN = "<!-- BEGIN REFINED_ONSET_SIGNAL_STATISTICS -->"
SECTION_END = "<!-- END REFINED_ONSET_SIGNAL_STATISTICS -->"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def finite_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def linear_quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def load_background_q95(metrics_path: Path, summary_path: Path) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in load_jsonl(metrics_path):
        method = str(row.get("method", ""))
        signal = str(row.get("signal", ""))
        value = finite_number(row.get("normalized_response_magnitude"))
        if method and signal and value is not None:
            groups[(method, signal)].append(value)
    computed = {}
    for key, values in groups.items():
        value = linear_quantile(values, 0.95)
        if value is not None:
            computed[key] = value

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("method", "")), str(row.get("signal", "")))
            reported = finite_number(row.get("normalized_response_magnitude_q95"))
            if key not in computed or reported is None:
                continue
            if not math.isclose(computed[key], reported, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Background Q95 mismatch for {key}: metrics={computed[key]} summary={reported}"
                )
    return computed


def direction(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, str, str]:
    method = str(row["method"])
    signal = str(row["signal"])
    outcome = str(row["outcome_group"])
    method_index = METHODS.index(method) if method in METHODS else len(METHODS)
    known_signals = SIGNALS_BY_METHOD.get(method, ())
    signal_index = known_signals.index(signal) if signal in known_signals else len(known_signals)
    outcome_index = OUTCOMES.index(outcome) if outcome in OUTCOMES else len(OUTCOMES)
    return method_index, signal_index, outcome_index, method, signal


def unique_event_rows(rows: list[dict[str, Any]], key: tuple[str, str, str]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        rollout_id = row.get("rollout_id")
        try:
            event_index = int(row["event_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing event identity in {key}: {row}") from error
        identity = (str(rollout_id), event_index)
        if identity in unique:
            raise ValueError(f"Duplicate event row for {key}: {identity}")
        unique[identity] = row
    return list(unique.values())


def fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def make_summary(event_rows: list[dict[str, Any]], background_q95: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        if bool(row.get("is_background")):
            continue
        key = (str(row.get("method", "")), str(row.get("signal", "")), str(row.get("outcome_group", "")))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        method, signal, outcome = key
        event_rows_unique = unique_event_rows(rows, key)
        q95 = background_q95.get((method, signal))
        if q95 is None:
            raise ValueError(f"No clean-background Q95 for {method}/{signal}")

        valid_magnitude = [
            row for row in event_rows_unique
            if finite_number(row.get("normalized_response_magnitude")) is not None
        ]
        valid_signed = [
            row for row in event_rows_unique
            if finite_number(row.get("signed_pre_post_change")) is not None
        ]
        exceeding = [
            row for row in valid_magnitude
            if float(row["normalized_response_magnitude"]) > q95
        ]

        all_signs = [direction(finite_number(row.get("signed_pre_post_change"))) for row in valid_signed]
        above_signs = [
            direction(finite_number(row.get("signed_pre_post_change")))
            for row in exceeding
            if finite_number(row.get("signed_pre_post_change")) is not None
        ]
        positive_all = all_signs.count("positive")
        negative_all = all_signs.count("negative")
        zero_all = all_signs.count("zero")
        positive_above = above_signs.count("positive")
        negative_above = above_signs.count("negative")
        zero_above = above_signs.count("zero")
        expected = EXPECTED_FAILURE_DIRECTION.get(method)
        consistency_all = sum(sign == expected for sign in all_signs)
        consistency_above = sum(sign == expected for sign in above_signs)

        summary.append({
            "method": method,
            "signal": signal,
            "outcome_group": outcome,
            "n_onset_events": len(event_rows_unique),
            "n_valid_response_magnitude_events": len(valid_magnitude),
            "clean_background_q95": q95,
            "n_exceeding_q95": len(exceeding),
            "exceeding_q95_fraction": fraction(len(exceeding), len(event_rows_unique)),
            "n_valid_signed_change_events": len(valid_signed),
            "signed_positive_count": positive_all,
            "signed_negative_count": negative_all,
            "signed_zero_count": zero_all,
            "signed_positive_count_above_q95": positive_above,
            "signed_negative_count_above_q95": negative_above,
            "signed_zero_count_above_q95": zero_above,
            "signed_positive_fraction_above_q95": fraction(positive_above, len(above_signs)),
            "signed_negative_fraction_above_q95": fraction(negative_above, len(above_signs)),
            "signed_zero_fraction_above_q95": fraction(zero_above, len(above_signs)),
            "expected_failure_direction": expected,
            "direction_consistency_count_all": consistency_all,
            "direction_consistency_fraction_all": fraction(consistency_all, len(event_rows_unique)),
            "direction_consistency_count_above_q95": consistency_above,
            "direction_consistency_fraction_above_q95": fraction(consistency_above, len(above_signs)),
        })
    return sorted(summary, key=sort_key)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt_number(value: Any, digits: int = 3) -> str:
    number = finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_percent(value: Any) -> str:
    number = finite_number(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


def fmt_count_rate(count: Any, rate: Any) -> str:
    return f"{int(count)}/{fmt_percent(rate)}"


def signed_counts(row: dict[str, Any], suffix: str = "") -> str:
    return "/".join(
        str(int(row[f"signed_{name}_count{suffix}"]))
        for name in ("positive", "negative", "zero")
    )


def signed_above_counts_rates(row: dict[str, Any]) -> str:
    counts = signed_counts(row, "_above_q95")
    rates = "/".join(
        fmt_percent(row[f"signed_{name}_fraction_above_q95"])
        for name in ("positive", "negative", "zero")
    )
    return f"{counts} ({rates})"


def build_section(rows: list[dict[str, Any]], metrics_path: Path, background_path: Path, summary_path: Path) -> str:
    lines = [
        SECTION_BEGIN,
        "## Refined onset threshold and direction statistics",
        "",
        "This section reuses the existing event-level and clean-success background metrics; no baseline inference was rerun.",
        "",
        "- Onset-associated change rate is n_exceeding_q95 / n_onset_events, using a strict > comparison against the clean-background Q95 of normalized_response_magnitude.",
        "- Signed directions are counted from raw signed_pre_post_change as positive / negative / zero. The above Q95 + / - / 0 field includes both counts and fractions among only the exceeding events.",
        "- Direction consistency is the fraction whose raw sign matches the expected failure direction: SAFE/RynnValue positive; ProcVLM/Robo-Dopamine negative. Zero changes are not consistent.",
        "- All current event rows have finite response magnitude and signed change. The CSV still records valid-event counts so denominators remain auditable.",
        f"- Sources: {metrics_path.name}, {background_path.name}, and validated {summary_path.name}.",
        "",
        "### Per method, signal, and outcome group",
        "",
        "| Method | Signal | Outcome | Onsets | Clean Q95 | Above Q95 (n/rate) | Signed + / - / 0 (all) | Above + / - / 0 (n and fraction) | Expected direction | Consistency all / above |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['signal']} | {row['outcome_group']} | "
            f"{row['n_onset_events']} | {fmt_number(row['clean_background_q95'])} | "
            f"{fmt_count_rate(row['n_exceeding_q95'], row['exceeding_q95_fraction'])} | "
            f"{signed_counts(row)} | {signed_above_counts_rates(row)} | "
            f"{row['expected_failure_direction']} | "
            f"{fmt_percent(row['direction_consistency_fraction_all'])} / {fmt_percent(row['direction_consistency_fraction_above_q95'])} |"
        )

    lines += [
        "",
        "### Terminal-failure vs recovered-success comparison",
        "",
        "The comparison below uses the same clean-background threshold and reports terminal minus recovered differences in percentage points.",
        "",
        "| Method | Signal | Terminal failure: n / change rate / consistency | Recovered success: n / change rate / consistency | Terminal - recovered: change rate / consistency |",
        "| --- | --- | --- | --- | ---: |",
    ]
    pairs = {(row["method"], row["signal"], row["outcome_group"]): row for row in rows}
    pair_keys = sorted(
        {
            (row["method"], row["signal"])
            for row in rows
            if row["outcome_group"] in {"terminal_failure", "recovered_success"}
        },
        key=lambda key: (
            METHODS.index(key[0]) if key[0] in METHODS else len(METHODS),
            SIGNALS_BY_METHOD.get(key[0], ()).index(key[1]) if key[1] in SIGNALS_BY_METHOD.get(key[0], ()) else len(SIGNALS_BY_METHOD.get(key[0], ())),
        ),
    )
    for method, signal in pair_keys:
        terminal = pairs.get((method, signal, "terminal_failure"))
        recovered = pairs.get((method, signal, "recovered_success"))
        if terminal is None or recovered is None:
            continue
        change_delta = (
            (float(terminal["exceeding_q95_fraction"]) - float(recovered["exceeding_q95_fraction"])) * 100.0
            if terminal["exceeding_q95_fraction"] is not None and recovered["exceeding_q95_fraction"] is not None
            else None
        )
        consistency_delta = (
            (float(terminal["direction_consistency_fraction_all"]) - float(recovered["direction_consistency_fraction_all"])) * 100.0
            if terminal["direction_consistency_fraction_all"] is not None and recovered["direction_consistency_fraction_all"] is not None
            else None
        )
        lines.append(
            f"| {method} | {signal} | "
            f"{terminal['n_onset_events']} / {fmt_percent(terminal['exceeding_q95_fraction'])} / {fmt_percent(terminal['direction_consistency_fraction_all'])} | "
            f"{recovered['n_onset_events']} / {fmt_percent(recovered['exceeding_q95_fraction'])} / {fmt_percent(recovered['direction_consistency_fraction_all'])} | "
            f"{fmt_number(change_delta, 1)} pp / {fmt_number(consistency_delta, 1)} pp |"
        )

    lines += [
        "",
        "Interpretation: a higher onset-associated change rate means more annotated onsets exceed the clean-success variability reference. Direction consistency is stricter than magnitude: a large response in the opposite raw direction is not counted as failure-consistent. Terminal/recovered differences are descriptive comparisons, not causal or detector-performance estimates.",
        SECTION_END,
    ]
    return "\n".join(lines)


def update_report(path: Path, section: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(SECTION_BEGIN) + r".*?" + re.escape(SECTION_END), re.DOTALL)
    if pattern.search(report):
        updated = pattern.sub(section, report, count=1)
    elif "## Artifacts" in report:
        updated = report.replace("## Artifacts", section + "\n\n## Artifacts", 1)
    else:
        updated = report.rstrip() + "\n\n" + section + "\n"
    path.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    event_path = analysis_dir / "event_metrics.jsonl"
    background_path = analysis_dir / "clean_background_metrics.jsonl"
    background_summary_path = analysis_dir / "clean_background_summary.csv"
    output_csv = (args.output_csv or analysis_dir / "onset_signal_statistics.csv").expanduser().resolve()
    report_path = (args.report or analysis_dir / "REPORT.md").expanduser().resolve()
    background_q95 = load_background_q95(background_path, background_summary_path)
    event_rows = load_jsonl(event_path)
    rows = make_summary(event_rows, background_q95)
    write_summary(output_csv, rows)
    section = build_section(rows, event_path, background_path, background_summary_path)
    update_report(report_path, section)
    print("REFINED_ONSET_SIGNAL_STATISTICS_OK")
    print(f"analysis_dir={analysis_dir}")
    print(f"summary_csv={output_csv}")
    print(f"groups={len(rows)}")
    print(f"event_rows={len(event_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
