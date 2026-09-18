from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .core import detect_hop_scale


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = PROJECT_ROOT / "tools" / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from robo_dopamine_multi_perspective import frame_index  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def ensure_within_project(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay inside PROJECT_ROOT: {resolved}"
        ) from exc
    return resolved


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Manifest contains duplicate rollout IDs: {path}")
    return result


def normalize_outcome(annotation: Mapping[str, Any]) -> str:
    outcome = annotation.get("outcome_label")
    if outcome == "failure":
        return "terminal_failure"
    if outcome == "recovered_success":
        return "recovered_success"
    if outcome == "success":
        return "clean_success"
    return str(outcome or "uncertain")


def load_annotation(
    annotation_dir: Path,
    rollout_id: str,
) -> dict[str, Any] | None:
    path = annotation_dir / f"{rollout_id}.json"
    if not path.is_file():
        return None
    annotation = load_json(path)
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation is not an object: {path}")

    events = list(annotation.get("failure_events") or [])
    if not events and annotation.get("observable_onset_frame") is not None:
        events = [
            {
                "failure_type": annotation.get("failure_type", "other"),
                "causal_onset_frame": annotation.get(
                    "causal_onset_frame"
                ),
                "observable_onset_frame": annotation.get(
                    "observable_onset_frame"
                ),
                "terminal_failure_frame": annotation.get(
                    "terminal_failure_frame"
                ),
                "recovery_frame": annotation.get("recovery_frame"),
                "notes": annotation.get("notes", ""),
            }
        ]

    clean_events: list[dict[str, Any]] = []
    for event_index, raw in enumerate(events):
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        event["event_index"] = event_index
        clean_events.append(event)
    annotation["failure_events"] = clean_events
    return annotation


def task_fields(
    manifest_row: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    suite = str(
        manifest_row.get("task_suite")
        or manifest_row.get("suite")
        or "unknown"
    )
    raw_task_id = manifest_row.get("task_id")
    task_id = "" if raw_task_id is None else str(raw_task_id)
    description = str(
        manifest_row.get("task_description")
        or manifest_row.get("task")
        or ""
    )
    if task_id:
        key = f"{suite}:task{task_id}"
    elif description:
        key = f"{suite}:{description}"
    else:
        key = suite
    return key, suite, task_id, description


def completed_rollout_ids(
    run_root: Path,
) -> tuple[list[str], list[str]]:
    paths = [run_root / "jobs.jsonl"]
    workers = run_root / "workers"
    if workers.is_dir():
        paths.extend(sorted(workers.glob("worker-*/jobs.jsonl")))

    completed: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        sources.append(project_relative(path))
        for row in load_jsonl(path):
            if row.get("status") != "complete":
                continue
            if row.get("return_code") not in (None, 0):
                continue
            rollout_id = row.get("rollout_id", row.get("id"))
            if (
                isinstance(rollout_id, str)
                and rollout_id
                and rollout_id not in seen
            ):
                seen.add(rollout_id)
                completed.append(rollout_id)
    return completed, sources


def relocate_recorded_path(
    value: str | Path,
    *,
    run_root: Path,
    worker_dir: Path,
) -> Path:
    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        parts = path.parts
        for anchor in (
            "outputs",
            "datasets",
            "annotations",
            "tools",
            "repos",
        ):
            if anchor in parts:
                index = parts.index(anchor)
                candidates.append(
                    PROJECT_ROOT.joinpath(*parts[index:])
                )
                break
    else:
        candidates.extend(
            (
                worker_dir / path,
                run_root / path,
                PROJECT_ROOT / path,
            )
        )

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return ensure_within_project(
                candidate, "incremental prediction"
            )
        except ValueError:
            # A saved absolute path may still exist on another checkout.
            # Ignore it and prefer a project-relative relocation candidate.
            continue

    for candidate in candidates:
        try:
            return ensure_within_project(
                candidate, "incremental prediction"
            )
        except ValueError:
            continue
    raise ValueError(
        "Recorded incremental prediction cannot be relocated inside "
        f"PROJECT_ROOT: {value}"
    )


def resolve_incremental_prediction(
    worker_result_path: Path,
    run_root: Path,
) -> tuple[Path, dict[str, Any], str]:
    result = load_json(worker_result_path)
    if not isinstance(result, dict):
        raise ValueError(
            f"Invalid worker result: {worker_result_path}"
        )

    recorded: Any = None
    source = ""
    perspectives = result.get("perspective_outputs")
    if isinstance(perspectives, dict):
        incremental = perspectives.get("incremental")
        if (
            isinstance(incremental, dict)
            and incremental.get("raw_model_output")
        ):
            recorded = incremental["raw_model_output"]
            source = (
                "worker_result.perspective_outputs.incremental"
            )

    if recorded is None:
        fusion = result.get("fusion")
        if isinstance(fusion, dict):
            paths = fusion.get("source_prediction_paths")
            if (
                isinstance(paths, dict)
                and paths.get("incremental")
            ):
                recorded = paths["incremental"]
                source = (
                    "worker_result.fusion."
                    "source_prediction_paths.incremental"
                )

    eval_mode = str(
        result.get("eval_mode") or ""
    ).lower()
    eval_modes = [
        str(value).lower()
        for value in (result.get("eval_modes") or [])
    ]
    if (
        recorded is None
        and (
            eval_mode == "incremental"
            or eval_modes == ["incremental"]
        )
    ):
        recorded = result.get("raw_model_output")
        source = "worker_result.raw_model_output"

    if recorded is None:
        metadata_path = (
            worker_result_path.parent
            / "multi_perspective"
            / "metadata.json"
        )
        if metadata_path.is_file():
            metadata = load_json(metadata_path)
            paths = (
                metadata.get("prediction_paths")
                if isinstance(metadata, dict)
                else None
            )
            if (
                isinstance(paths, dict)
                and paths.get("incremental")
            ):
                recorded = paths["incremental"]
                source = (
                    "multi_perspective.metadata."
                    "prediction_paths.incremental"
                )

    if recorded is None:
        raise FileNotFoundError(
            "Completed Robo-Dopamine rollout has no saved "
            f"incremental perspective: {worker_result_path}"
        )

    prediction = relocate_recorded_path(
        recorded,
        run_root=run_root,
        worker_dir=worker_result_path.parent,
    )
    if not prediction.is_file():
        raise FileNotFoundError(
            "Recorded incremental pred_vllm.json does not exist: "
            f"{prediction}"
        )
    return prediction, result, source


def load_incremental_signal(
    run_root: Path,
    rollout_id: str,
) -> dict[str, Any]:
    worker_result_path = (
        run_root / "raw" / rollout_id / "worker_result.json"
    )
    if not worker_result_path.is_file():
        raise FileNotFoundError(worker_result_path)

    prediction_path, worker_result, path_source = (
        resolve_incremental_prediction(
            worker_result_path, run_root
        )
    )
    raw_rows = load_json(prediction_path)
    if not isinstance(raw_rows, list):
        raise ValueError(
            "Robo-Dopamine incremental pred_vllm.json "
            f"is not a list: {prediction_path}"
        )

    rows: list[dict[str, Any]] = []
    frames: list[int] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError(
                f"Robo-Dopamine row is not an object: {prediction_path}"
            )
        rows.append(raw)
        frames.append(frame_index(raw))

    if len(frames) != len(set(frames)):
        raise ValueError(
            f"Duplicate native frame indices in {prediction_path}"
        )

    order = sorted(range(len(frames)), key=frames.__getitem__)
    frames = [frames[index] for index in order]
    rows = [rows[index] for index in order]
    scale = detect_hop_scale(rows)

    return {
        "rollout_id": rollout_id,
        "frames": frames,
        "raw_hops": list(scale["raw_hops"]),
        "hops": list(scale["normalized_hops"]),
        "source_scale": scale["source_scale"],
        "normalization_divisor": scale[
            "normalization_divisor"
        ],
        "scale_detection": {
            key: value
            for key, value in scale.items()
            if key not in {"raw_hops", "normalized_hops"}
        },
        "prediction_path": prediction_path,
        "prediction_path_source": path_source,
        "worker_result_path": worker_result_path,
        "frame_interval": worker_result.get("frame_interval"),
        "source_commit": worker_result.get("source_commit"),
        "checkpoint": worker_result.get("checkpoint"),
        "eval_mode": worker_result.get("eval_mode"),
        "eval_modes": worker_result.get("eval_modes"),
    }


def build_base_records(
    run_root: Path,
    manifest: Mapping[str, Mapping[str, Any]],
    annotation_dir: Path,
    allowed_rollout_ids: set[str] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    completed_ids, jobs_sources = completed_rollout_ids(
        run_root
    )
    completed_before_filter = len(completed_ids)
    if allowed_rollout_ids is not None:
        completed_ids = [
            rollout_id
            for rollout_id in completed_ids
            if rollout_id in allowed_rollout_ids
        ]
    if not completed_ids:
        raise ValueError(
            f"No completed Robo-Dopamine jobs found under {run_root}"
        )

    signals: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    clean_rollouts: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    scale_counter: Counter[str] = Counter()
    frame_intervals: Counter[str] = Counter()
    source_commits: Counter[str] = Counter()
    checkpoints: Counter[str] = Counter()

    for rollout_id in completed_ids:
        manifest_row = manifest.get(rollout_id)
        if manifest_row is None:
            exclusions.append(
                {
                    "rollout_id": rollout_id,
                    "reason": "manifest_missing",
                }
            )
            continue

        annotation = load_annotation(
            annotation_dir, rollout_id
        )
        if annotation is None:
            exclusions.append(
                {
                    "rollout_id": rollout_id,
                    "reason": "annotation_missing",
                }
            )
            continue

        try:
            signal = load_incremental_signal(
                run_root, rollout_id
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            exclusions.append(
                {
                    "rollout_id": rollout_id,
                    "reason": (
                        "incremental_output_unavailable: "
                        + str(exc)
                    ),
                }
            )
            continue

        if not signal["frames"]:
            exclusions.append(
                {
                    "rollout_id": rollout_id,
                    "reason": "incremental_output_empty",
                }
            )
            continue

        task_key, suite, task_id, description = (
            task_fields(manifest_row)
        )
        outcome = normalize_outcome(annotation)
        signal.update(
            {
                "task_key": task_key,
                "task_suite": suite,
                "task_id": task_id,
                "task_description": description,
                "outcome": outcome,
            }
        )
        signals[rollout_id] = signal

        scale_counter[str(signal["source_scale"])] += 1
        frame_intervals[str(signal.get("frame_interval"))] += 1
        if signal.get("source_commit"):
            source_commits[
                str(signal["source_commit"])
            ] += 1
        if signal.get("checkpoint"):
            checkpoints[str(signal["checkpoint"])] += 1

        if outcome == "clean_success":
            clean_rollouts.append(
                {
                    "rollout_id": rollout_id,
                    "task_key": task_key,
                    "task_suite": suite,
                    "task_id": task_id,
                    "task_description": description,
                    "outcome": outcome,
                }
            )

        for event in annotation.get(
            "failure_events", []
        ):
            onset = event.get(
                "observable_onset_frame"
            )
            if onset is None:
                continue
            try:
                onset_frame = int(onset)
            except (TypeError, ValueError):
                continue

            def optional_int(name: str) -> int | None:
                value = event.get(name)
                return (
                    int(value)
                    if value is not None
                    else None
                )

            events.append(
                {
                    "event_id": (
                        f"{rollout_id}::event"
                        f"{int(event.get('event_index', 0))}"
                    ),
                    "rollout_id": rollout_id,
                    "event_index": int(
                        event.get("event_index", 0)
                    ),
                    "failure_type": str(
                        event.get("failure_type")
                        or annotation.get("failure_type")
                        or "other"
                    ),
                    "observable_onset_frame": onset_frame,
                    "causal_onset_frame": optional_int(
                        "causal_onset_frame"
                    ),
                    "recovery_frame": optional_int(
                        "recovery_frame"
                    ),
                    "terminal_failure_frame": optional_int(
                        "terminal_failure_frame"
                    ),
                    "outcome": outcome,
                    "task_key": task_key,
                    "task_suite": suite,
                    "task_id": task_id,
                    "task_description": description,
                }
            )

    if not signals:
        raise ValueError(
            "No completed rollouts have usable saved "
            "incremental-hop outputs"
        )
    if len(scale_counter) > 1:
        raise ValueError(
            "Mixed incremental-hop storage scales in one "
            f"analysis run: {dict(scale_counter)}"
        )

    provenance = {
        "completed_rollout_n": len(completed_ids),
        "completed_rollout_n_before_selection": completed_before_filter,
        "selection_filter_n": (
            len(allowed_rollout_ids)
            if allowed_rollout_ids is not None
            else None
        ),
        "usable_rollout_n": len(signals),
        "event_n": len(events),
        "clean_rollout_n": len(clean_rollouts),
        "excluded_rollouts": exclusions,
        "jobs_sources": jobs_sources,
        "hop_source_scales": dict(scale_counter),
        "frame_intervals": dict(frame_intervals),
        "robo_source_commits": dict(source_commits),
        "checkpoints": dict(checkpoints),
    }
    return signals, events, clean_rollouts, provenance
