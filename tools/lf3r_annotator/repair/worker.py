#!/usr/bin/env python3
"""Execute the Repair worker with Ctrl-World-safe provenance finalization.

The implementation lives in worker_core.py.  Ctrl-World intentionally does not
load the recorded LIBERO action array at runtime, so provenance must derive the
action horizon from the already validated alignment contract instead of calling
len() on A2World-only arrays.
"""

from __future__ import annotations

from pathlib import Path


_BAD = '''                "gt_action_end": int(len(actions)),
                "gt_future_action_count": int(len(future_actions)),'''

_FIXED = '''                "gt_action_end": int(
                    len(actions)
                    if actions is not None
                    else alignment["gt_action_end"]
                ),
                "gt_future_action_count": int(
                    len(future_actions)
                    if future_actions is not None
                    else max(
                        0,
                        int(alignment["gt_action_end"])
                        - int(alignment["gt_action_start"]),
                    )
                ),'''


def main() -> None:
    entrypoint = Path(__file__).resolve()
    implementation = entrypoint.with_name("worker_core.py")
    source = implementation.read_text(encoding="utf-8")
    if source.count(_BAD) != 1:
        raise RuntimeError(
            "Repair worker provenance patch no longer matches worker_core.py; "
            "update the entrypoint together with the implementation"
        )
    patched = source.replace(_BAD, _FIXED, 1)
    namespace = {
        "__name__": "__main__",
        "__file__": str(entrypoint),
        "__package__": None,
    }
    exec(compile(patched, str(entrypoint), "exec"), namespace)


if __name__ == "__main__":
    main()
