#!/usr/bin/env python3
"""Select and download diverse failed DROID RGB trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "droid_subset"
BUCKET = "gresearch"
OBJECT_PREFIX = "robotics/droid_raw/1.0.1/"
API_ROOT = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
PUBLIC_ROOT = f"https://storage.googleapis.com/{BUCKET}/"
METADATA_GLOB = "**/failure/**/metadata_*.json"


def get_json(url: str, *, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LF3R-dataset-subset/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise RuntimeError(f"DROID metadata request failed {attempts} times: {url}") from last_error


def list_failure_metadata() -> list[dict]:
    objects: list[dict] = []
    page_token: str | None = None
    while True:
        params = {
            "prefix": OBJECT_PREFIX,
            "matchGlob": METADATA_GLOB,
            "maxResults": "1000",
            "fields": "items(name,size),nextPageToken",
        }
        if page_token:
            params["pageToken"] = page_token
        url = API_ROOT + "?" + urllib.parse.urlencode(params)
        page = get_json(url)
        objects.extend(page.get("items", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    objects.sort(key=lambda item: item["name"])
    return objects


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if len(items) <= count:
        return items
    if count <= 1:
        return [items[len(items) // 2]]
    indexes = {round(i * (len(items) - 1) / (count - 1)) for i in range(count)}
    return [items[index] for index in sorted(indexes)]


def read_metadata(object_name: str) -> dict:
    url = PUBLIC_ROOT + urllib.parse.quote(object_name, safe="/")
    return get_json(url)


def normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def choose_diverse(candidates: list[dict], count: int) -> list[dict]:
    pool = sorted(candidates, key=lambda item: item["object_name"])
    chosen: list[dict] = []
    tasks: set[str] = set()
    scenes: set[str] = set()
    buildings: set[str] = set()
    labs: set[str] = set()
    dates: set[str] = set()

    while pool and len(chosen) < count:
        def score(item: dict) -> tuple[int, int, int, int, int]:
            meta = item["metadata"]
            task = normalize(meta.get("current_task"))
            scene = str(meta.get("scene_id") or "")
            building = normalize(meta.get("building"))
            lab = normalize(meta.get("lab"))
            date = str(meta.get("date") or "")
            return (
                int(bool(task) and task not in tasks),
                int(bool(scene) and scene not in scenes),
                int(bool(building) and building not in buildings),
                int(bool(lab) and lab not in labs),
                int(bool(date) and date not in dates),
            )

        best = max(pool, key=score)
        pool.remove(best)
        chosen.append(best)
        meta = best["metadata"]
        tasks.add(normalize(meta.get("current_task")))
        scenes.add(str(meta.get("scene_id") or ""))
        buildings.add(normalize(meta.get("building")))
        labs.add(normalize(meta.get("lab")))
        dates.add(str(meta.get("date") or ""))
    return chosen


def head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "LF3R-dataset-subset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError(f"No Content-Length returned for {url}")
    return int(value)


def download_resumable(url: str, destination: Path, expected_size: int) -> tuple[int, bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_size:
            return 0, True
        raise RuntimeError(f"Existing file has an unexpected size; preserving it: {destination}")

    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Partial file is larger than the published object: {partial}")
    transferred = 0
    consecutive_failures = 0

    while not partial.exists() or partial.stat().st_size < expected_size:
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {"User-Agent": "LF3R-dataset-subset/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            before = offset
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError("Object server did not honor the resume range")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        output.flush()
                        transferred += len(block)
            actual = partial.stat().st_size
            if actual > expected_size:
                raise RuntimeError(f"Downloaded more bytes than expected: {partial}")
            if actual == expected_size:
                break
            if actual > before:
                consecutive_failures = 0
            else:
                raise IOError(f"Download ended without progress at byte {actual}")
        except Exception as exc:
            after = partial.stat().st_size if partial.exists() else 0
            if after > offset:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= 3:
                raise RuntimeError(f"DROID stopped after three consecutive download failures: {url}") from exc
            time.sleep(consecutive_failures or 1)

    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete DROID object: {partial}")
    partial.replace(destination)
    return transferred, False


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=240)
    args = parser.parse_args()
    if args.limit < 1 or args.candidate_limit < args.limit:
        parser.error("--limit must be positive and --candidate-limit must be at least --limit")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    print("Inspecting the public DROID raw bucket failure metadata index...", flush=True)
    metadata_objects = list_failure_metadata()
    print(f"Found {len(metadata_objects)} failure metadata objects.", flush=True)
    sampled = evenly_spaced(metadata_objects, args.candidate_limit)
    candidates: list[dict] = []
    for index, item in enumerate(sampled, start=1):
        metadata = read_metadata(item["name"])
        if metadata.get("success") is False and metadata.get("ext1_mp4_path"):
            candidates.append({"object_name": item["name"], "metadata": metadata})
        if index % 40 == 0 or index == len(sampled):
            print(f"Read {index}/{len(sampled)} stratified episode metadata records.", flush=True)

    if len(candidates) < args.limit:
        raise RuntimeError(f"Only {len(candidates)} usable failed episodes were found; need {args.limit}.")
    selected = choose_diverse(candidates, args.limit)
    records: list[dict] = []
    for item in selected:
        meta = item["metadata"]
        lab = meta.get("lab") or item["object_name"].split("/")[3]
        object_name = f"{OBJECT_PREFIX}{lab}/{meta['ext1_mp4_path']}"
        video_url = PUBLIC_ROOT + urllib.parse.quote(object_name, safe="/")
        episode_id = meta.get("uuid") or Path(item["object_name"]).stem.removeprefix("metadata_")
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", episode_id)
        records.append(
            {
                "episode_id": episode_id,
                "episode_index": None,
                "task": meta.get("current_task"),
                "scene": {"building": meta.get("building"), "scene_id": meta.get("scene_id")},
                "source": {"release": "DROID raw 1.0.1", "lab": lab, "date": meta.get("date"), "metadata_object": item["object_name"]},
                "camera_view": "ext1_mp4_path (external camera 1; single RGB left-view video)",
                "video_object": object_name,
                "video_path": f"videos/{safe_id}_ext1.mp4",
                "video_url": video_url,
            }
        )

    sizes: list[int] = []
    for record in records:
        record["size_bytes"] = head_size(record["video_url"])
        sizes.append(record["size_bytes"])
    required = sum(sizes) + 100 * 1024 * 1024
    available = shutil.disk_usage(DATASET_ROOT).free
    print(f"Selected {len(records)} episodes across {len({normalize(r['task']) for r in records})} task labels and {len({str(r['scene'].get('scene_id')) for r in records})} scene IDs.", flush=True)
    print(f"Planned external-view download: {sum(sizes) / (1024**2):.1f} MiB; free space: {available / (1024**3):.1f} GiB.", flush=True)
    if available < required:
        raise RuntimeError(f"Insufficient disk space: need at least {required} bytes, have {available}.")

    manifest = {
        "dataset": "DROID",
        "source": "https://droid-dataset.github.io/droid/the-droid-dataset",
        "selection_method": "Stratified sample of failure metadata, greedily prioritizing unseen task labels, scene IDs, buildings, labs, and collection dates.",
        "camera_view": "ext1: external camera 1, single RGB view",
        "episodes": records,
    }
    manifest_path = DATASET_ROOT / "selected_episodes.json"
    write_manifest(manifest_path, manifest)

    transferred = 0
    reused = 0
    for index, record in enumerate(records, start=1):
        destination = DATASET_ROOT / record["video_path"]
        amount, was_reused = download_resumable(record["video_url"], destination, record["size_bytes"])
        transferred += amount
        reused += int(was_reused)
        record["downloaded_bytes"] = record["size_bytes"]
        record["download_status"] = "reused" if was_reused else "downloaded"
        write_manifest(manifest_path, manifest)
        print(f"DROID {index}/{len(records)}: {record['episode_id']} ({record['size_bytes'] / (1024**2):.1f} MiB)", flush=True)

    disk_bytes = sum((DATASET_ROOT / record["video_path"]).stat().st_size for record in records)
    print(json.dumps({"episodes": len(records), "tasks": len({normalize(r['task']) for r in records}), "scenes": len({str(r['scene'].get('scene_id')) for r in records}), "camera": manifest["camera_view"], "disk_bytes": disk_bytes, "transferred_bytes": transferred, "reused_files": reused, "path": str(DATASET_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
