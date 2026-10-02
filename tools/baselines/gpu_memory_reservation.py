"""Hold CUDA memory with PyTorch until the vLLM worker requests handoff."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import uuid

SOCKET_ENV = "LF3R_ROBO_RESERVATION_SOCKET"
TOKEN_ENV = "LF3R_ROBO_RESERVATION_TOKEN"
MIB = 1024 * 1024


def release_reserved_device(device: int) -> None:
    path = os.environ.get(SOCKET_ENV)
    if not path:
        return
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(60)
        connection.connect(path)
        connection.sendall((json.dumps({"token": os.environ.get(TOKEN_ENV), "device": device}) + "\n").encode())
        response = json.loads(connection.makefile("rb").readline())
    if not response.get("released"):
        raise RuntimeError(f"GPU reservation handoff failed: {response}")
    print(f"LF3R_GPU_RESERVATION_RELEASED device={device}", flush=True)


class MemoryReservation:
    """The runner owns the keeper's stdin; EOF releases memory on owner exit."""

    def __init__(self, python: Path, project_root: Path, env: dict[str, str], log_path: Path,
                 fraction: float, devices: int, reserve_mib: int = 0, handoff_buffer_mib: int = 2048):
        folder = project_root / "cache/tmp"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / ("gpu-reserve-" + uuid.uuid4().hex + ".sock")
        self.token = secrets.token_hex(32)
        self.env = dict(env)
        self.process = None
        self.log_path = log_path
        self.command = [str(python), str(Path(__file__).resolve()),
                        "--socket", str(self.path), "--token", self.token,
                        "--fraction", str(fraction), "--devices", str(devices),
                        "--reserve-mib", str(reserve_mib), "--handoff-buffer-mib", str(handoff_buffer_mib)]
        self.budget = None

    def start(self):
        with self.log_path.open("a", encoding="utf-8") as errors:
            self.process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in self.process.stdout:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "gpu_reservation_failed":
                    raise RuntimeError("GPU reservation failed: " + event["error"])
                if event.get("event") == "gpu_reservation_ready":
                    self.budget = event["budget"]
                    return self.budget
            raise RuntimeError(f"GPU reservation failed (exit {self.process.wait()}); see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def worker_env(self):
        search_path = str(Path(__file__).resolve().parent)
        if self.env.get("PYTHONPATH"):
            search_path += os.pathsep + self.env["PYTHONPATH"]
        return {**self.env, "PYTHONPATH": search_path, SOCKET_ENV: str(self.path), TOKEN_ENV: self.token,
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

    def close(self):
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.process.stdout is not None:
                self.process.stdout.close()
        self.path.unlink(missing_ok=True)


def hold_memory(torch, fraction: float, devices: int, reserve_mib: int = 0, handoff_buffer_mib: int = 2048):
    handoff_bytes = handoff_buffer_mib * MIB
    if devices < 1 or devices > torch.cuda.device_count():
        raise ValueError("Reservation requires at least tensor-parallel-size visible CUDA devices")
    snapshots = []
    for device in range(devices):
        with torch.cuda.device(device):
            torch.cuda.init()
            free, total = torch.cuda.mem_get_info()
            target = reserve_mib * MIB - handoff_bytes if reserve_mib else min(int(free * fraction), free - 2048 * MIB - handoff_bytes)
            if target <= 0 or target + handoff_bytes > free - 2048 * MIB:
                raise ValueError(f"GPU {device}: insufficient free memory for reservation plus 2048 MiB headroom")
            snapshots.append((free, total, target))
    # vLLM uses the same total-memory fraction on all tensor-parallel ranks.
    resolved = math.floor(min(target / total for _, total, target in snapshots) * 1_000_000) / 1_000_000
    allocations = {}
    try:
        for device, (free, total, _) in enumerate(snapshots):
            with torch.cuda.device(device):
                allocations[device] = torch.empty(int(total * resolved) + handoff_bytes, dtype=torch.uint8, device=f"cuda:{device}")
                torch.cuda.synchronize()
    except BaseException:
        allocations.clear()
        for device in range(devices):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        raise
    return allocations, {"scope": "reserved_gpu_memory", "resolved_total_fraction": resolved,
                         "resolution": "torch_reservation_before_model_import",
                         "safety_buffer_mib_per_gpu": 2048,
                         "handoff_buffer_mib_per_gpu": handoff_buffer_mib,
                         "reserved_mib_per_gpu": [(int(total * resolved) + handoff_bytes) / MIB for _, total, _ in snapshots],
                         "vllm_target_mib_per_gpu": [int(total * resolved) / MIB for _, total, _ in snapshots]}


def keeper_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--reserve-mib", type=int, default=0)
    parser.add_argument("--handoff-buffer-mib", type=int, default=2048)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.reserve_mib < 0 or args.handoff_buffer_mib < 0:
        parser.error("fraction must be in (0,1] and reserve-mib non-negative")
    print(json.dumps({"event": "gpu_reservation_importing_torch"}), flush=True)
    # Exit even during a slow import if the runner died and its pipe closed.
    def owner_watch():
        while os.read(sys.stdin.fileno(), 4096):
            pass
        os._exit(0)
    threading.Thread(target=owner_watch, daemon=True).start()
    import torch
    allocations, budget = hold_memory(torch, args.fraction, args.devices, args.reserve_mib, args.handoff_buffer_mib)
    path = Path(args.socket)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(args.devices)
            print(json.dumps({"event": "gpu_reservation_ready", "budget": budget}), flush=True)
            while allocations:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(10)
                    try:
                        request = json.loads(connection.makefile("rb").readline(4096))
                        device = request.get("device")
                        if not secrets.compare_digest(str(request.get("token", "")), args.token):
                            raise ValueError("invalid handoff token")
                        if type(device) is not int or device not in allocations:
                            raise ValueError("device has no active reservation")
                        with torch.cuda.device(device):
                            del allocations[device]
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        response = {"released": True, "device": device}
                    except (ValueError, OSError) as error:
                        response = {"released": False, "error": str(error)}
                    connection.sendall((json.dumps(response) + "\n").encode())
    finally:
        allocations.clear()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        keeper_main()
    except Exception as error:
        print(json.dumps({"event": "gpu_reservation_failed", "error": str(error)}), flush=True)
        os._exit(1)
    # This dedicated process owns only transient CUDA memory. All buffers and
    # the socket were released above; avoid PyTorch/Python finalizer stalls.
    os._exit(0)
