#!/usr/bin/env python3
"""LF3R annotator entrypoint with multi-manifest support and raw video serving.

Builds on ``server_entry_v3`` so multi-manifest catalog support and all prior
WebUI extensions remain intact, but deliberately disables runtime video codec
probing/transcoding. The video endpoint always serves the exact manifest file.

If a source codec is not browser-decodable, convert that dataset/video offline
instead of changing the bytes behind a live Range URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import server_entry_v3


def _raw_request_path(self: Any, path: Path) -> Path:
    """Always serve the original manifest video unchanged."""
    return path


def _raw_status(self: Any, path: Path) -> dict[str, Any]:
    """Compatibility endpoint remains harmless for stale clients."""
    return {
        "status": "disabled",
        "mode": "raw_source",
        "message": "Runtime video transcoding is disabled; serving the manifest source unchanged.",
    }


server_entry_v3.AsyncVideoCompatibilityCache.request_path = _raw_request_path
server_entry_v3.AsyncVideoCompatibilityCache.status = _raw_status


if __name__ == "__main__":
    server_entry_v3.main()
