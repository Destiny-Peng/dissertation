#!/usr/bin/env python3
"""Compare already-generated Robo-Dopamine multi-mode sanity runs.

This report intentionally does not run inference.  Run the same selected
rollouts with ``--robo-frame-interval 2``, ``5`` and ``10`` (each with
``--robo-eval-modes incremental forward backward``), then pass the resulting
run roots here.  Keeping the report separate makes the interval comparison
reproducible without loading another checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from robo_dopamine_multi_perspective import PERSPECTIVE_MODES, summarize_curve


def load_records(run_root: Path) -> list[dict[str, Any]]:
    records = []
    for result_path in sorted((run_root / "raw").glob("*/worker_result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("multi_perspective") is not True:
            continue
        perspectives = result.get("perspective_outputs") or {}
        if set(perspectives) != set(PERSPECTIVE_MODES):
            raise ValueError(f"Incomplete multi-perspective result: {result_path}")
        curves = {
            mode: summarize_curve(Path(perspectives[mode]["raw_model_output"]), mode)
            for mode in PERSPECTIVE_MODES
        }
        fused = summarize_curve(Path(result["fused_model_output"]), "fused")
        records.append(
            {
                "run_root": str(run_root),
                "rollout_id": result_path.parent.name,
                "frame_interval": int(result["frame_interval"]),
                "curves": {**curves, "fused": fused},
                "mode_seconds": result.get("mode_seconds", {}),
            }
        )
    return records


def write_report(run_roots: list[Path], output_dir: Path, expected: list[int]) -> dict[str, Any]:
    records = [record for root in run_roots for record in load_records(root)]
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "expected_intervals": expected,
        "observed_intervals": sorted({record["frame_interval"] for record in records}),
        "records": records,
        "note": "Descriptive smoothness only; interval=2 noise is not a detector-performance conclusion.",
    }
    (output_dir / "interval_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = [
        "rollout_id",
        "frame_interval",
        "mode",
        "samples",
        "diff_std",
        "mean_abs_diff",
        "sign_change_rate",
        "range",
    ]
    with (output_dir / "interval_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for mode, summary in record["curves"].items():
                writer.writerow(
                    {
                        "rollout_id": record["rollout_id"],
                        "frame_interval": record["frame_interval"],
                        "mode": mode,
                        **{field: summary.get(field) for field in fields[3:]},
                    }
                )

    # A compact plot is useful for the specific interval=2 question, but the
    # CSV/JSON remain authoritative and do not require matplotlib.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        report["plot_path"] = None
        return report
    grouped: dict[tuple[int, str], list[float]] = {}
    for record in records:
        for mode, summary in record["curves"].items():
            grouped.setdefault((record["frame_interval"], mode), []).append(float(summary["diff_std"]))
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=140)
    for mode in (*PERSPECTIVE_MODES, "fused"):
        xs = sorted(interval for interval, name in grouped if name == mode)
        if xs:
            ys = [sum(grouped[(interval, mode)]) / len(grouped[(interval, mode)]) for interval in xs]
            axis.plot(xs, ys, marker="o", label=mode)
    axis.set_xlabel("frame interval")
    axis.set_ylabel("std of progress differences")
    axis.set_title("Robo-Dopamine interval sanity: incremental noise and curve stability")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    plot_path = output_dir / "interval_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    report["plot_path"] = str(plot_path)
    (output_dir / "interval_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-interval", action="append", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = write_report(args.run_root, args.output_dir, args.expected_interval or [2, 5, 10])
    print(json.dumps({key: result[key] for key in ("expected_intervals", "observed_intervals", "plot_path") if key in result}, ensure_ascii=False))

