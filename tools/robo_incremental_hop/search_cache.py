"""Persistent content-addressed cache for expensive fused-hop detector search."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import PROJECT_ROOT

SEARCH_CACHE_SCHEMA = 1
SEARCH_SEMANTICS_VERSION = "fused-phenotype-operational-v3-oracle-localization-v2"
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
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
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
