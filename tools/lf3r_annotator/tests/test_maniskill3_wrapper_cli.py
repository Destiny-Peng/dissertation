from __future__ import annotations

import ast
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = TOOL_ROOT / "generate_maniskill3_success.py"


def test_wrapper_does_not_delegate_cli_parsing_to_upstream() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(WRAPPER))
    assert "maniskill_run.parse_args(" not in source
    assert "def _upstream_args(" in source
    assert "start_seed=args.seed" in source
    assert 'config["width"] = args.render_width' in source
    assert 'config["height"] = args.render_height' in source
    assert "LF3R_MANISKILL3_START=" in source
