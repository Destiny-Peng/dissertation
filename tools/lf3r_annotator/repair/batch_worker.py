#!/usr/bin/env python3
"""Execute one Repair multi-run group sequentially on one selected GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _update_status(group_dir: Path, **updates: Any) -> dict[str, Any]:
    path = group_dir / "status.json"
    payload = _read_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(path, payload)
    return payload


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"Repair item path escapes PROJECT_ROOT: {value}") from error
    return resolved


def _safe_log_name(index: int, rollout_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", rollout_id).strip("-")[:80] or "rollout"
    return f"{index:03d}-{safe}.log"


def _run_and_tee(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as item_log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            item_log.write(line)
            item_log.flush()
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--group-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    group_dir = args.group_dir.resolve()
    try:
        group_dir.relative_to(project_root)
    except ValueError as error:
        raise SystemExit("group-dir must be inside PROJECT_ROOT") from error

    plan = json.loads((group_dir / "plan.json").read_text(encoding="utf-8"))
    items = plan.get("items") or []
    if not isinstance(items, list) or not items:
        raise SystemExit("Repair run group has no items")

    worker = Path(__file__).resolve().parent / "worker.py"
    completed = 0
    failed = 0
    results: list[dict[str, Any]] = []
    total = len(items)
    _update_status(
        group_dir,
        status="running",
        phase=f"0/{total} complete · 0 failed",
        progress=0.0,
        expected_runs=total,
        completed_runs=0,
        failed_runs=0,
        current_rollout=None,
        current_run_id=None,
        error=None,
    )

    try:
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                failed += 1
                continue

            rollout_id = str(item.get("rollout_id") or "")
            run_id = str(item.get("run_id") or "")
            run_dir = _project_path(project_root, str(item.get("run_dir") or ""))
            worker_python_path = Path(str(item.get("worker_python") or "")).expanduser()
            worker_python = (
                worker_python_path.absolute()
                if plan.get("world_model") == "wan2_2"
                else worker_python_path.resolve()
            )
            item_log = group_dir / _safe_log_name(index, rollout_id)
            item["log_path"] = str(item_log.relative_to(project_root))
            _atomic_json(group_dir / "plan.json", plan)

            print(
                f"[repair-group] {index}/{total} start "
                f"rollout={rollout_id} run={run_id} log={item_log}",
                flush=True,
            )
            _update_status(
                group_dir,
                status="running",
                phase=(
                    f"running {index}/{total} · "
                    f"{completed} complete · {failed} failed"
                ),
                progress=float((index - 1) / total),
                current_rollout=rollout_id,
                current_run_id=run_id,
                current_log=str(item_log.relative_to(project_root)),
                completed_runs=completed,
                failed_runs=failed,
            )

            environment = os.environ.copy()
            module_root = str(Path(__file__).resolve().parents[1])
            existing = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                module_root if not existing else module_root + os.pathsep + existing
            )
            return_code = _run_and_tee(
                [
                    str(worker_python),
                    str(worker),
                    "--project-root",
                    str(project_root),
                    "--run-dir",
                    str(run_dir),
                ],
                cwd=project_root,
                environment=environment,
                log_path=item_log,
            )
            run_status = _read_json(run_dir / "status.json", {})
            succeeded = bool(
                return_code == 0
                and isinstance(run_status, dict)
                and run_status.get("status") == "complete"
            )
            if succeeded:
                completed += 1
                result_status = "complete"
            else:
                failed += 1
                result_status = "failed"

            results.append(
                {
                    "rollout_id": rollout_id,
                    "run_id": run_id,
                    "status": result_status,
                    "return_code": return_code,
                    "log_path": str(item_log.relative_to(project_root)),
                    "error": (
                        run_status.get("error")
                        if isinstance(run_status, dict)
                        else None
                    ),
                }
            )
            _atomic_json(group_dir / "results.json", results)
            done = completed + failed
            print(
                f"[repair-group] {index}/{total} {result_status} "
                f"rollout={rollout_id} run={run_id} rc={return_code}",
                flush=True,
            )
            _update_status(
                group_dir,
                status="running",
                phase=f"{done}/{total} complete · {failed} failed",
                progress=float(done / total),
                current_rollout=None,
                current_run_id=None,
                current_log=None,
                completed_runs=completed,
                failed_runs=failed,
            )

        terminal = (
            "complete"
            if failed == 0
            else ("complete_with_errors" if completed > 0 else "failed")
        )
        _update_status(
            group_dir,
            status=terminal,
            phase=f"{completed}/{total} complete · {failed} failed",
            progress=1.0,
            current_rollout=None,
            current_run_id=None,
            current_log=None,
            completed_runs=completed,
            failed_runs=failed,
            completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            error=(
                None
                if terminal == "complete"
                else f"{failed} of {total} Repair runs failed"
            ),
        )
        print(
            f"[repair-group] finished status={terminal} "
            f"complete={completed} failed={failed} total={total}",
            flush=True,
        )
    except Exception as error:
        _update_status(
            group_dir,
            status="failed",
            phase="failed",
            progress=float((completed + failed) / total),
            completed_runs=completed,
            failed_runs=failed,
            current_rollout=None,
            current_run_id=None,
            current_log=None,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
