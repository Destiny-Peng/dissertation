"""Read-only analysis service composition."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analysis_details import AnalysisDetailsMixin
from analysis_snapshots import AnalysisSnapshotsMixin


class AnalysisService(
    AnalysisSnapshotsMixin,
    AnalysisDetailsMixin,
):
    """Expose analysis snapshots, dashboard details, and artifacts."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path,
        annotation_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.annotation_root = annotation_root.resolve()
        self.analysis_root = (
            self.project_root
            / "outputs"
            / "baseline_signal_analysis"
        )
        self.robo_hop_root = (
            self.project_root
            / "outputs"
            / "robo_dopamine_incremental_hop"
        )
        self.robo_localization_head_root = (
            self.project_root
            / "outputs"
            / "robo_dopamine_localization_head"
        )
        self.robo_label_loss_root = (
            self.project_root
            / "outputs"
            / "robo_dopamine_label_loss_ablation"
        )

    def _relative(self, path: Path) -> str:
        try:
            return str(
                path.resolve().relative_to(
                    self.project_root
                )
            )
        except ValueError:
            return str(path)

    @staticmethod
    def _coerce(value: Any) -> Any:
        if isinstance(
            value,
            (list, dict, tuple),
        ):
            return None
        if value in (None, ""):
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                number = float(value)
            except ValueError:
                return value
            if not math.isfinite(number):
                return None
            return (
                int(number)
                if number.is_integer()
                else number
            )
        if (
            isinstance(value, float)
            and not math.isfinite(value)
        ):
            return None
        return value

    def _read_csv(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        with path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            return [
                {
                    key: self._coerce(value)
                    for key, value
                    in row.items()
                }
                for row in csv.DictReader(handle)
            ]

    def _manifest(
        self,
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        with self.manifest_path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[row["id"]] = row
        return records

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _iso_mtime(path: Path) -> str:
        return dt.datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=dt.timezone.utc,
        ).isoformat()
