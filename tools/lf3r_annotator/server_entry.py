#!/usr/bin/env python3
"""Stable LF3R WebUI entrypoint.

The WebUI is composed from semantic feature modules instead of versioned
server_entry_vN layers. Keep this file as the only supported executable entry.
"""

from __future__ import annotations

import webui_runtime


def main() -> None:
    webui_runtime.main()


if __name__ == "__main__":
    main()
