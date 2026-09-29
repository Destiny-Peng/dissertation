#!/usr/bin/env python3
"""Download and extract diverse REBOOT recovery episodes from official shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "reboot_subset"
CAMERA_KEY = "observation.images.cam_high"
CAMERA_COLUMN = f"videos/{CAMERA_KEY}"
REPOS = [
    ("han10e_recovery_install", "HAN 10E", "install"),
    ("han10e_recovery_remove", "HAN 10E", "remove"),
    ("m12_recovery_install", "M12 fastener", "install"),
    ("m12_recovery_remove", "M12 fastener", "remove"),
    ("16mm-cylinder_recovery_install", "16 mm cylinder", "install"),
    ("16mm-cylinder_recovery_remove", "16 mm cylinder", "remove"),
    ("16mm-bar_recovery_install", "16 mm bar", "install"),
    ("rca_recovery_install", "RCA connector", "install"),
    ("16mm-bar_recovery_remove", "16 mm bar", "remove"),
    ("USBC_recovery_remove", "USB-C", "remove"),
    ("nema1-15-plug_recovery_install", "NEMA 1-15P", "install"),
    ("nema1-15-plug_recovery_remove2", "NEMA 1-15P", "remove"),
    ("rj45_recovery_install", "RJ45", "install"),
    ("RJ45_recovery_remove", "RJ45", "remove"),
    ("USB-A_recovery_install", "USB-A", "install"),
]
TARGET_INDICES = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]


def request_bytes(url: str, *, headers: dict | None = None, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request_headers = {"User-Agent": "LF3R-dataset-subset/1.0"}
            if headers:
                request_headers.update(headers)
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"REBOOT request failed three times: {url}") from last_error


def request_json(url: str) -> dict:
    return json.loads(request_bytes(url))


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_pyarrow():
    if sys.version_info[:2] == (3, 11):
        sys.path.insert(0, str(PROJECT_ROOT / "cache" / "dataset-runtime"))
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Use LF3R's Python 3.11 environment with PyArrow available to read the official episode index.") from exc
    return pa, pq


def download_resumable(url: str, destination: Path, expected_size: int, expected_sha256: str | None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"Existing source shard has an unexpected size; preserving it: {destination}")
        if expected_sha256 and sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"Existing source shard checksum is wrong; preserving it: {destination}")
        return 0
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Partial source shard is larger than expected: {partial}")

    transferred = 0
    consecutive_failures = 0
    while not partial.exists() or partial.stat().st_size < expected_size:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        try:
            request_headers = {"User-Agent": "LF3R-dataset-subset/1.0"}
            if headers:
                request_headers.update(headers)
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                if offset and response.status != 206:
                    raise IOError("Hugging Face file server did not honor the resume range")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        output.flush()
                        transferred += len(block)
            actual = partial.stat().st_size
            if actual > expected_size:
                raise RuntimeError(f"Downloaded more bytes than the published source shard size: {partial}")
            if actual == expected_size:
                break
            if actual > offset:
                consecutive_failures = 0
            else:
                raise IOError(f"Source shard ended at byte {actual} without progress")
        except Exception as exc:
            actual = partial.stat().st_size if partial.exists() else 0
            if actual > offset:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= 3:
                raise RuntimeError("REBOOT stopped after three consecutive video-shard download failures") from exc
            time.sleep(consecutive_failures or 1)

    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete REBOOT video shard: {partial}")
    if expected_sha256 and sha256_file(partial) != expected_sha256:
        raise RuntimeError(f"REBOOT shard checksum mismatch: {partial}")
    partial.replace(destination)
    return transferred


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def valid_clip(path: Path, expected_duration: float) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    try:
        return abs(float(result.stdout.strip()) - expected_duration) <= 0.2
    except ValueError:
        return False


def make_clip(source: Path, destination: Path, start: float, end: float) -> None:
    duration = end - start
    if duration <= 0:
        raise RuntimeError(f"Invalid REBOOT episode timestamps {start}–{end}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".part.mp4")
    if partial.exists():
        partial.unlink()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.9f}", "-i", str(source), "-t", f"{duration:.9f}",
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-movflags", "+faststart", str(partial),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not extract {destination.name}: {result.stderr[-1200:]}")
    if not valid_clip(partial, duration):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Extracted clip duration check failed for {destination.name}")
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    if args.limit != len(REPOS):
        parser.error(f"This curated diverse subset currently contains {len(REPOS)} task repositories; use --limit {len(REPOS)}")

    pa, pq = load_pyarrow()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required to extract individual episodes from video shards.")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    discovered: list[dict] = []
    source_shards: dict[tuple[str, str], dict] = {}
    print("Inspecting official REBOOT task repositories, info.json, and episode timing metadata...", flush=True)

    for order, (repo_name, object_name, direction) in enumerate(REPOS):
        repo = f"REBOOT26/{repo_name}"
        repo_info = request_json(f"https://huggingface.co/api/datasets/{repo}")
        revision = repo_info.get("sha") or "main"
        tree_url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
        tree = request_json(tree_url)
        files = {item["path"]: item for item in tree if item.get("type") == "file"}
        if "meta/info.json" not in files:
            raise RuntimeError(f"Official info.json is missing for {repo}")
        info_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/meta/info.json?download=true"
        info = json.loads(request_bytes(info_url))
        camera_feature = info.get("features", {}).get(f"observation.images.{CAMERA_KEY.removeprefix('observation.images.')}", {})
        if camera_feature.get("dtype") != "video":
            raise RuntimeError(f"The selected main RGB camera is unavailable in {repo}")

        episode_paths = sorted(path for path in files if path.startswith("meta/episodes/") and path.endswith(".parquet"))
        if not episode_paths:
            raise RuntimeError(f"Official episode timing metadata is missing for {repo}")
        rows: list[dict] = []
        columns = ["episode_index", "tasks", "length"]
        columns.extend(
            [
                f"{CAMERA_COLUMN}/chunk_index",
                f"{CAMERA_COLUMN}/file_index",
                f"{CAMERA_COLUMN}/from_timestamp",
                f"{CAMERA_COLUMN}/to_timestamp",
            ]
        )
        for path in episode_paths:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path, safe='/')}?download=true"
            data = request_bytes(url)
            table = pq.read_table(pa.BufferReader(data), columns=columns)
            rows.extend(table.to_pylist())
        if not rows:
            raise RuntimeError(f"The official episode table is empty for {repo}")

        target_index = TARGET_INDICES[order]
        selected_row = min(rows, key=lambda row: abs(int(row["episode_index"]) - target_index))
        chunk_index = int(selected_row[f"{CAMERA_COLUMN}/chunk_index"])
        file_index = int(selected_row[f"{CAMERA_COLUMN}/file_index"])
        shard_path = f"{CAMERA_COLUMN}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        shard_meta = files.get(shard_path)
        if shard_meta is None:
            raise RuntimeError(f"Camera shard {shard_path} is absent from {repo}")
        shard_size = int(shard_meta.get("lfs", {}).get("size", shard_meta.get("size", 0)))
        shard_sha = shard_meta.get("lfs", {}).get("oid")
        shard_key = (repo, shard_path)
        source_shards[shard_key] = {
            "repo": repo,
            "revision": revision,
            "path": shard_path,
            "size_bytes": shard_size,
            "sha256": shard_sha,
            "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(shard_path, safe='/')}?download=true",
        }
        task_list = selected_row.get("tasks") or []
        task = str(task_list[0]) if task_list else f"{object_name} {direction}"
        episode_index = int(selected_row["episode_index"])
        duration = float(selected_row[f"{CAMERA_COLUMN}/to_timestamp"]) - float(selected_row[f"{CAMERA_COLUMN}/from_timestamp"])
        slug = repo_name.replace("/", "_")
        output_rel = f"videos/{slug}_episode_{episode_index:04d}_cam_high.mp4"
        discovered.append(
            {
                "episode_id": f"{repo}#episode-{episode_index}",
                "episode_index": episode_index,
                "task": task,
                "scene": {"object": object_name, "direction": direction, "setup": "bi-manual WidowX precision-assembly station"},
                "source": {"dataset_repository": repo, "revision": revision, "recovery_collection": repo_name},
                "camera_view": "cam_high (overhead RGB)",
                "video_path": output_rel,
                "source_shard": shard_path,
                "source_shard_size_bytes": shard_size,
                "source_shard_sha256": shard_sha,
                "source_timestamp_start": float(selected_row[f"{CAMERA_COLUMN}/from_timestamp"]),
                "source_timestamp_end": float(selected_row[f"{CAMERA_COLUMN}/to_timestamp"]),
                "duration_seconds": duration,
                "frame_count": int(selected_row["length"]),
                "robot_type": info.get("robot_type"),
                "_shard_key": shard_key,
            }
        )
        print(f"Selected {repo} episode {episode_index} from {shard_path} ({shard_size / (1024**2):.1f} MiB).", flush=True)

    source_total = sum(item["size_bytes"] for item in source_shards.values())
    clip_reserve = len(discovered) * 50 * 1024 * 1024
    required = source_total + clip_reserve + 100 * 1024 * 1024
    available = shutil.disk_usage(DATASET_ROOT).free
    print(f"Selected {len(discovered)} recovery trajectories across {len({item['scene']['object'] for item in discovered})} objects and {len({item['task'] for item in discovered})} task labels.", flush=True)
    print(f"Selected camera source shards total {source_total / (1024**3):.2f} GiB; free space: {available / (1024**3):.1f} GiB.", flush=True)
    if available < required:
        raise RuntimeError(f"Insufficient disk space: need at least {required} bytes, have {available}.")

    for item in discovered:
        item.pop("_shard_key")
    manifest = {
        "dataset": "REBOOT",
        "source": "https://nanayawoa.github.io/REBOOT/",
        "license": "CC-BY-4.0 (official project page)",
        "selection_method": "One recovery episode from each of 15 curated task repositories, covering all nine assembly objects and both install/remove directions where released.",
        "camera_view": "cam_high (overhead RGB); no depth or other camera views downloaded",
        "episodes": discovered,
    }
    manifest_path = DATASET_ROOT / "selected_episodes.json"
    write_manifest(manifest_path, manifest)

    transferred = 0
    reused = 0
    for order, item in enumerate(discovered):
        shard_key = (item["source"]["dataset_repository"], item["source_shard"])
        shard = source_shards[shard_key]
        output = DATASET_ROOT / item["video_path"]
        if valid_clip(output, item["duration_seconds"]):
            reused += 1
            item["download_status"] = "reused"
            item["video_size_bytes"] = output.stat().st_size
            write_manifest(manifest_path, manifest)
            print(f"REBOOT {order + 1}/{len(discovered)}: reused {output.name}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f"Existing clip failed validation; preserving it: {output}")

        source_path = DATASET_ROOT / ".staging" / f"{item['source']['dataset_repository'].split('/')[-1]}__{Path(shard['path']).name}"
        amount = download_resumable(shard["url"], source_path, shard["size_bytes"], shard["sha256"])
        transferred += amount
        start = item["source_timestamp_start"]
        end = item["source_timestamp_end"]
        make_clip(source_path, output, start, end)
        item["download_status"] = "downloaded"
        item["source_shard_transfer_bytes"] = amount
        item["video_size_bytes"] = output.stat().st_size
        write_manifest(manifest_path, manifest)
        source_path.unlink(missing_ok=True)
        print(f"REBOOT {order + 1}/{len(discovered)}: {output.name} ({item['video_size_bytes'] / (1024**2):.1f} MiB)", flush=True)

    disk_bytes = sum((DATASET_ROOT / item["video_path"]).stat().st_size for item in discovered)
    print(json.dumps({"episodes": len(discovered), "tasks": len({item['task'] for item in discovered}), "objects": len({item['scene']['object'] for item in discovered}), "camera": manifest["camera_view"], "source_shard_bytes_planned": source_total, "source_bytes_transferred": transferred, "video_disk_bytes": disk_bytes, "reused_clips": reused, "path": str(DATASET_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
