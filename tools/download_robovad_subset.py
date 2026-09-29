#!/usr/bin/env python3
"""Range-extract a diverse anomalous RoboVAD video subset."""

from __future__ import annotations

import argparse
import binascii
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import re
import shutil
import struct
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "robovad_subset"
RECORD_ID = "22754659"
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/RoboVAD.zip/content"
RANGE_CHUNK = 256 * 1024


def request_json(url: str, *, attempts: int = 3) -> dict:
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
    raise RuntimeError(f"RoboVAD metadata request failed {attempts} times: {url}") from last_error


class RangeReader:
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self._cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError(f"Invalid seek mode: {whence}")
        if self.position < 0:
            raise ValueError("Negative seek position")
        return self.position

    def _fetch(self, start: int, length: int) -> bytes:
        end = start + length - 1
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "LF3R-dataset-subset/1.0"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise IOError(f"Range request returned HTTP {response.status}")
                    data = response.read()
                if len(data) != length:
                    raise IOError(f"Short range: received {len(data)} of {length} bytes")
                return data
            except Exception as exc:
                last_error = exc
                if attempt + 1 < 3:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"RoboVAD range request failed three times at byte {start}") from last_error

    def read_at(self, offset: int, length: int) -> bytes:
        if offset >= self.size or length <= 0:
            return b""
        length = min(length, self.size - offset)
        output = bytearray()
        cursor = offset
        while len(output) < length:
            block_start = (cursor // RANGE_CHUNK) * RANGE_CHUNK
            block = self._cache.get(block_start)
            if block is None:
                block_length = min(RANGE_CHUNK, self.size - block_start)
                block = self._fetch(block_start, block_length)
                self._cache[block_start] = block
            block_offset = cursor - block_start
            amount = min(len(block) - block_offset, length - len(output))
            output.extend(block[block_offset : block_offset + amount])
            cursor += amount
        return bytes(output)

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self.position
        data = self.read_at(self.position, length)
        self.position += len(data)
        return data


def normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def task_for(episode_name: str) -> str:
    if episode_name.startswith("h_pickandplace_multi_"):
        return "multiple-cube pick-and-place"
    if episode_name.startswith("h_pickandplace_"):
        return "single-cube pick-and-place"
    if episode_name.startswith("h_stackcubes_"):
        return "cube stacking"
    if episode_name.startswith("h_pour_rice_"):
        return "rice pouring"
    if episode_name.startswith("h_rings_on_peg_"):
        return "ring insertion"
    return "unknown"


def episode_index(episode_name: str) -> int | None:
    match = re.search(r"_ep(\d+)$", episode_name)
    return int(match.group(1)) if match else None


def choose_diverse(candidates: list[dict], count: int) -> list[dict]:
    remaining = sorted(candidates, key=lambda item: item["episode_id"])
    chosen: list[dict] = []
    seen_tasks: set[str] = set()
    seen_types: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_severities: set[str] = set()

    while remaining and len(chosen) < count:
        def score(item: dict) -> tuple[int, int, int, int, int]:
            task = item["task"]
            types = set(item["anomaly_types"])
            pairs = {(task, event_type) for event_type in types}
            severities = set(item["severities"])
            return (
                len(pairs - seen_pairs),
                len(types - seen_types),
                int(task not in seen_tasks),
                len(severities - seen_severities),
                len(types),
            )

        best = max(remaining, key=score)
        remaining.remove(best)
        chosen.append(best)
        seen_tasks.add(best["task"])
        seen_types.update(best["anomaly_types"])
        seen_pairs.update((best["task"], event_type) for event_type in best["anomaly_types"])
        seen_severities.update(best["severities"])
    return chosen


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stream_zip_member(reader: RangeReader, info: zipfile.ZipInfo, raw_path: Path) -> int:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and raw_path.stat().st_size > info.compress_size:
        raise RuntimeError(f"Staged ZIP member exceeds its published compressed size: {raw_path}")
    before = raw_path.stat().st_size if raw_path.exists() else 0
    transferred = 0
    consecutive_failures = 0

    while not raw_path.exists() or raw_path.stat().st_size < info.compress_size:
        offset = raw_path.stat().st_size if raw_path.exists() else 0
        data_start_header = reader.read_at(info.header_offset, 30)
        if len(data_start_header) != 30 or data_start_header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local ZIP header for {info.filename}")
        name_length, extra_length = struct.unpack_from("<HH", data_start_header, 26)
        member_start = info.header_offset + 30 + name_length + extra_length
        start = member_start + offset
        end = member_start + info.compress_size - 1
        request = urllib.request.Request(
            reader.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "LF3R-dataset-subset/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise IOError(f"Archive server did not honor range for {info.filename}")
                mode = "ab" if offset else "wb"
                with raw_path.open(mode) as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        output.flush()
                        transferred += len(block)
            actual = raw_path.stat().st_size
            if actual > info.compress_size:
                raise RuntimeError(f"Staged ZIP member is larger than expected: {raw_path}")
            if actual == info.compress_size:
                break
            if actual > offset:
                consecutive_failures = 0
            else:
                raise IOError(f"No bytes received for {info.filename}")
        except Exception as exc:
            actual = raw_path.stat().st_size if raw_path.exists() else 0
            if actual > offset:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= 3:
                raise RuntimeError("RoboVAD stopped after three consecutive video download failures") from exc
            time.sleep(consecutive_failures or 1)

    if raw_path.stat().st_size != info.compress_size:
        raise RuntimeError(f"Incomplete ZIP member: {raw_path}")
    return transferred


def extract_verified(info: zipfile.ZipInfo, raw_path: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite an existing video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    crc = 0
    size = 0
    try:
        with raw_path.open("rb") as source, partial.open("wb") as output:
            decompressor = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise RuntimeError(f"Unsupported ZIP compression method {info.compress_type} for {info.filename}")
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                decoded = block if decompressor is None else decompressor.decompress(block)
                if decoded:
                    output.write(decoded)
                    size += len(decoded)
                    crc = binascii.crc32(decoded, crc)
            if decompressor is not None:
                tail = decompressor.flush()
                if tail:
                    output.write(tail)
                    size += len(tail)
                    crc = binascii.crc32(tail, crc)
        if size != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
            raise RuntimeError(f"ZIP member integrity check failed for {info.filename}")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    record = request_json(RECORD_API)
    archive_record = next((item for item in record.get("files", []) if item.get("key") == "RoboVAD.zip"), None)
    if archive_record is None:
        raise RuntimeError("RoboVAD.zip is missing from the official Zenodo record.")
    archive_size = int(archive_record["size"])
    request = urllib.request.Request(ARCHIVE_URL, method="HEAD", headers={"User-Agent": "LF3R-dataset-subset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        remote_size = int(response.headers["Content-Length"])
    if remote_size != archive_size:
        raise RuntimeError(f"Zenodo record and archive sizes differ: {archive_size} != {remote_size}")

    print("Inspecting RoboVAD archive directory and its small annotation/split metadata by HTTP ranges...", flush=True)
    reader = RangeReader(ARCHIVE_URL, archive_size)
    with zipfile.ZipFile(reader) as archive:
        entries = {info.filename: info for info in archive.infolist()}
        annotations = json.loads(archive.read("RoboVAD/annotations.json"))
        test_rows = csv.DictReader(io.StringIO(archive.read("RoboVAD/splits/test.csv").decode("utf-8-sig")))
        test_ids = {row["episode_name"] for row in test_rows}
        readme = archive.read("RoboVAD/README.md").decode("utf-8", errors="replace")
        license_match = re.search(r"CC BY[^\n]*", readme, re.IGNORECASE)
        license_text = license_match.group(0).strip(" .") if license_match else "See the source dataset README."

        candidates: list[dict] = []
        for episode_id, events in annotations.items():
            if episode_id not in test_ids:
                continue
            video_name = f"RoboVAD/videos/{episode_id}_left.mp4"
            if video_name not in entries:
                continue
            task = task_for(episode_id)
            anomaly_types = sorted({str(event.get("type", "unknown")) for event in events})
            severities = sorted({str(event.get("categories", {}).get("severity", "unknown")) for event in events})
            candidates.append(
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index(episode_id),
                    "task": task,
                    "scene": "standard tabletop robot-manipulation setup; no per-episode scene ID is released",
                    "source": {"record_id": RECORD_ID, "split": "test", "archive_member": video_name},
                    "anomaly_types": anomaly_types,
                    "anomaly_events": events,
                    "severities": severities,
                    "camera_view": "left static external RGB camera",
                    "video_path": f"videos/{episode_id}_left.mp4",
                    "archive_member": video_name,
                    "_zip_info": entries[video_name],
                }
            )

        if len(candidates) < args.limit:
            raise RuntimeError(f"Only {len(candidates)} anomalous test episodes have left-camera video; need {args.limit}.")
        selected = choose_diverse(candidates, args.limit)
        transfer_estimate = sum(item["_zip_info"].compress_size + item["_zip_info"].file_size for item in selected)
        needed = transfer_estimate + 100 * 1024 * 1024
        available = shutil.disk_usage(DATASET_ROOT).free
        tasks_covered = len({item["task"] for item in selected})
        types_covered = len({event_type for item in selected for event_type in item["anomaly_types"]})
        print(f"Selected {len(selected)} test anomalies across {tasks_covered} tasks and {types_covered} anomaly types.", flush=True)
        print(f"Conservative selected-file space estimate: {transfer_estimate / (1024**2):.1f} MiB; free space: {available / (1024**3):.1f} GiB.", flush=True)
        if available < needed:
            raise RuntimeError(f"Insufficient disk space: need at least {needed} bytes, have {available}.")

        for item in selected:
            item["size_bytes"] = item["_zip_info"].file_size
            item["compressed_size_bytes"] = item["_zip_info"].compress_size
            del item["_zip_info"]
        manifest = {
            "dataset": "RoboVAD",
            "source": record.get("links", {}).get("self_html", f"https://zenodo.org/records/{RECORD_ID}"),
            "record_checksum": archive_record.get("checksum"),
            "license": license_text,
            "selection_method": "Greedy coverage of unseen task/anomaly pairs, anomaly types, tasks, and severities among annotated test-split episodes.",
            "camera_view": "left static external RGB camera",
            "episodes": selected,
        }
        manifest_path = DATASET_ROOT / "selected_episodes.json"
        write_manifest(manifest_path, manifest)

        transferred = 0
        reused = 0
        total = len(selected)

        def download_one(item: dict) -> tuple[int, str]:
            info = entries[item["archive_member"]]
            destination = DATASET_ROOT / item["video_path"]
            if destination.exists():
                if destination.stat().st_size != info.file_size:
                    raise RuntimeError(f"Existing video has an unexpected size; preserving it: {destination}")
                return 0, "reused"
            raw_path = DATASET_ROOT / ".staging" / f"{item['episode_id']}_left.zipmember.part"
            amount = stream_zip_member(reader, info, raw_path)
            extract_verified(info, raw_path, destination)
            raw_path.unlink(missing_ok=True)
            return amount, "downloaded"

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_one, item): item for item in selected}
            completed = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    amount, status = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                completed += 1
                transferred += amount
                reused += int(status == "reused")
                item["download_status"] = status
                item["downloaded_archive_bytes"] = amount
                write_manifest(manifest_path, manifest)
                info = entries[item["archive_member"]]
                print(f"RoboVAD {completed}/{total}: {item['episode_id']} ({info.file_size / (1024**2):.1f} MiB)", flush=True)

    disk_bytes = sum((DATASET_ROOT / item["video_path"]).stat().st_size for item in selected)
    print(json.dumps({"episodes": len(selected), "tasks": tasks_covered, "anomaly_types": types_covered, "camera": manifest["camera_view"], "disk_bytes": disk_bytes, "archive_bytes_transferred": transferred, "reused_files": reused, "path": str(DATASET_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
