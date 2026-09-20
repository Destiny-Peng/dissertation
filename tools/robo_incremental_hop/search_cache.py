"""Persistent content-addressed cache for expensive fused-hop detector search."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import PROJECT_ROOT

SEARCH_CACHE_SCHEMA = 1
SEARCH_SEMANTICS_VERSION = (
    "fused-phenotype-operational-v3-oracle-localization-v2-ensemble-sweep-v1"
)
SEARCH_CACHE_ROOT = PROJECT_ROOT / "cache" / "robo_dopamine_hop_search"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # Stable enough for values already parsed from saved JSON outputs.
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def search_fingerprint(
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
) -> str:
    """Hash only inputs that can change detector-search outcomes."""
    signal_payload = [
        {
            "rollout_id": rollout_id,
            "signal_source": str(signal.get("prediction_path") or ""),
            "frames": [int(value) for value in signal["frames"]],
            "hops": [float(value) for value in signal["hops"]],
        }
        for rollout_id, signal in sorted(signals.items())
    ]
    payload = {
        "schema": SEARCH_CACHE_SCHEMA,
        "search_semantics_version": SEARCH_SEMANTICS_VERSION,
        "signals": signal_payload,
        "events": _jsonable(events),
        "no_event_failures": _jsonable(no_event_failures),
        "clean_rollouts": _jsonable(clean_rollouts),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(fingerprint: str) -> Path:
    if len(fingerprint) != 64 or any(
        char not in "0123456789abcdef" for char in fingerprint
    ):
        raise ValueError("invalid search fingerprint")
    return SEARCH_CACHE_ROOT / f"{fingerprint}.json.gz"


def load_search_cache(fingerprint: str) -> dict[str, Any] | None:
    path = cache_path(fingerprint)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("schema") != SEARCH_CACHE_SCHEMA:
        return None
    if document.get("search_semantics_version") != SEARCH_SEMANTICS_VERSION:
        return None
    if document.get("fingerprint") != fingerprint:
        return None
    required = (
        "configs",
        "oracle_configs",
        "phenotype_grid",
        "summary_rows",
        "event_rows",
        "no_event_rows",
        "clean_rows",
        "ensemble_sweep",
        "oracle_global_best",
        "oracle_event_detectability",
        "oracle_summary",
    )
    if any(key not in document for key in required):
        return None
    return document


def write_search_cache(
    fingerprint: str,
    *,
    configs: Sequence[Mapping[str, Any]],
    oracle_configs: Sequence[Mapping[str, Any]],
    phenotype_grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    no_event_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
    ensemble_sweep: Sequence[Mapping[str, Any]],
    oracle_global_best: Sequence[Mapping[str, Any]],
    oracle_event_detectability: Sequence[Mapping[str, Any]],
    oracle_summary: Sequence[Mapping[str, Any]],
) -> Path:
    path = cache_path(fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SEARCH_CACHE_SCHEMA,
        "search_semantics_version": SEARCH_SEMANTICS_VERSION,
        "fingerprint": fingerprint,
        "configs": _jsonable(configs),
        "oracle_configs": _jsonable(oracle_configs),
        "phenotype_grid": _jsonable(phenotype_grid),
        "summary_rows": _jsonable(summary_rows),
        "event_rows": _jsonable(event_rows),
        "no_event_rows": _jsonable(no_event_rows),
        "clean_rows": _jsonable(clean_rows),
        "ensemble_sweep": _jsonable(ensemble_sweep),
        "oracle_global_best": _jsonable(oracle_global_best),
        "oracle_event_detectability": _jsonable(oracle_event_detectability),
        "oracle_summary": _jsonable(oracle_summary),
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{fingerprint}.",
        suffix=".json.gz.tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path



def _typed_csv_value(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if value.strip() == value and all(
            token not in value.lower() for token in (".", "e", "nan", "inf")
        ):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _read_csv_typed(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                key: _typed_csv_value(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def _config_signature(configs: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    fields = ("config_id", "detector_family", "parameters_json")
    return sorted(
        tuple(config.get(field) for field in fields)
        for config in configs
    )


def _configs_from_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        config_id = row.get("config_id")
        if config_id in (None, ""):
            continue
        key = str(config_id)
        result.setdefault(
            key,
            {
                "config_id": key,
                "detector_family": row.get("detector_family"),
                "parameters_json": row.get("parameters_json"),
            },
        )
    return list(result.values())


def _event_descriptor(row: Mapping[str, Any]) -> tuple[Any, ...]:
    event_id = row.get("event_id")
    if event_id in (None, ""):
        event_id = f"{row.get('rollout_id')}::event{int(row.get('event_index') or 0)}"
    return (
        str(event_id),
        str(row.get("rollout_id") or ""),
        int(row.get("event_index") or 0),
        str(row.get("failure_type") or ""),
        str(row.get("outcome") or ""),
        int(row.get("observable_onset_frame") or 0),
        (
            int(row["episode_end_frame"])
            if row.get("episode_end_frame") is not None
            else None
        ),
        str(row.get("episode_end_source") or ""),
        str(row.get("task_key") or ""),
    )


def _rollout_descriptor(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("rollout_id") or ""),
        str(row.get("outcome") or ""),
        str(row.get("task_key") or ""),
    )


def _unique_descriptors(
    rows: Sequence[Mapping[str, Any]],
    descriptor: Any,
) -> list[tuple[Any, ...]]:
    return sorted({descriptor(row) for row in rows})


def _metadata_time(metadata: Mapping[str, Any]) -> float | None:
    value = metadata.get("generated_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _inputs_older_than(
    generated_at: float,
    *,
    signals: Mapping[str, Mapping[str, Any]],
    annotation_dir: Path,
) -> bool:
    tolerance = 1.0
    for signal in signals.values():
        path = signal.get("prediction_path")
        if path is None:
            return False
        try:
            if Path(path).stat().st_mtime > generated_at + tolerance:
                return False
        except OSError:
            return False
    try:
        for path in annotation_dir.rglob("*"):
            if path.is_file() and path.stat().st_mtime > generated_at + tolerance:
                return False
    except OSError:
        return False
    return True


def load_legacy_search_seed(
    output_root: Path,
    *,
    current_output_dir: Path,
    run_root_relative: str,
    manifest_sha256: str,
    annotation_dir: Path,
    signals: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    no_event_failures: Sequence[Mapping[str, Any]],
    clean_rollouts: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    oracle_configs: Sequence[Mapping[str, Any]],
    phenotype_grid: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, Path | None]:
    """Best-effort import of a compatible pre-cache completed analysis.

    This is intentionally conservative. It requires the same run root, manifest,
    rollout set, empirical detector grid, event/no-event/clean denominators, and
    no input file newer than the candidate analysis timestamp.
    """
    required = (
        "metadata.json",
        "sweep_summary.csv",
        "event_results.csv",
        "no_event_failure_results.csv",
        "clean_rollout_results.csv",
        "ensemble_sweep.csv",
        "oracle_global_best.csv",
        "oracle_event_detectability.csv",
        "oracle_summary.csv",
    )
    if not output_root.is_dir():
        return None, None

    candidates: list[tuple[float, Path]] = []
    for metadata_path in output_root.glob("*/metadata.json"):
        directory = metadata_path.parent
        if directory.resolve() == current_output_dir.resolve():
            continue
        if not all((directory / name).is_file() for name in required):
            continue
        try:
            candidates.append((metadata_path.stat().st_mtime, directory))
        except OSError:
            continue

    current_event_descriptors = sorted(_event_descriptor(row) for row in events)
    current_no_event_descriptors = sorted(
        _rollout_descriptor(row) for row in no_event_failures
    )
    current_clean_descriptors = sorted(
        _rollout_descriptor(row) for row in clean_rollouts
    )
    expected_grid = _jsonable(phenotype_grid)
    expected_rollouts = sorted(signals)
    current_sources = {
        rollout_id: str(signal.get("prediction_path") or "")
        for rollout_id, signal in signals.items()
    }

    for _mtime, directory in sorted(candidates, reverse=True):
        try:
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        input_meta = metadata.get("input") or {}
        signal_meta = metadata.get("signal") or {}
        if input_meta.get("run_root") != run_root_relative:
            continue
        if input_meta.get("manifest_sha256") != manifest_sha256:
            continue
        if sorted(signal_meta.get("common_rollout_ids") or []) != expected_rollouts:
            continue

        previous_grid = dict(metadata.get("phenotype_detector") or {})
        previous_grid.pop("clean_fpr_constraints", None)
        if _jsonable(previous_grid) != expected_grid:
            continue

        oracle_meta = metadata.get("oracle_analysis") or {}
        if oracle_meta.get("early_tolerance_native_samples") != 1:
            continue

        generated_at = _metadata_time(metadata)
        if generated_at is None or not _inputs_older_than(
            generated_at,
            signals=signals,
            annotation_dir=annotation_dir,
        ):
            continue

        try:
            summary_rows = _read_csv_typed(directory / "sweep_summary.csv")
            event_rows = _read_csv_typed(directory / "event_results.csv")
            no_event_rows = _read_csv_typed(
                directory / "no_event_failure_results.csv"
            )
            clean_rows = _read_csv_typed(directory / "clean_rollout_results.csv")
            ensemble_sweep = _read_csv_typed(directory / "ensemble_sweep.csv")
            oracle_global_best = _read_csv_typed(
                directory / "oracle_global_best.csv"
            )
            oracle_event_detectability = _read_csv_typed(
                directory / "oracle_event_detectability.csv"
            )
            oracle_summary = _read_csv_typed(directory / "oracle_summary.csv")
        except (OSError, csv.Error):
            continue

        if _config_signature(_configs_from_summary(summary_rows)) != _config_signature(configs):
            continue
        if _unique_descriptors(event_rows, _event_descriptor) != current_event_descriptors:
            continue
        if _unique_descriptors(no_event_rows, _rollout_descriptor) != current_no_event_descriptors:
            continue
        if _unique_descriptors(clean_rows, _rollout_descriptor) != current_clean_descriptors:
            continue

        previous_sources: dict[str, str] = {}
        for row in [*event_rows, *no_event_rows, *clean_rows]:
            rollout_id = str(row.get("rollout_id") or "")
            source = row.get("signal_source")
            if rollout_id and isinstance(source, str):
                previous_sources.setdefault(rollout_id, source)
        # Prior CSV stores project-relative source paths; compare suffixes so
        # project-root relocation does not invalidate an otherwise exact run.
        if any(
            not previous_sources.get(rollout_id)
            or not current_sources[rollout_id].endswith(previous_sources[rollout_id])
            for rollout_id in expected_rollouts
        ):
            continue

        return (
            {
                "configs": [dict(config) for config in configs],
                "oracle_configs": [dict(config) for config in oracle_configs],
                "phenotype_grid": dict(phenotype_grid),
                "summary_rows": summary_rows,
                "event_rows": event_rows,
                "no_event_rows": no_event_rows,
                "clean_rows": clean_rows,
                "ensemble_sweep": ensemble_sweep,
                "oracle_global_best": oracle_global_best,
                "oracle_event_detectability": oracle_event_detectability,
                "oracle_summary": oracle_summary,
            },
            directory,
        )

    return None, None
