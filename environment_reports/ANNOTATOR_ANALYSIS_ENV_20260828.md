# LF3R Annotator Analysis Environment — 2026-08-28

## Status

The temporal analysis environment is ready and was validated without running
GPU inference.

| Item | Value |
| --- | --- |
| Project-local environment | `conda_envs/LF3R-ananlyse` |
| Creation tool | `uv venv` (the host has no `conda`) |
| Python | `3.10.21` |
| Analysis interpreter | `conda_envs/LF3R-ananlyse/bin/python` |
| Matplotlib backend | `Agg` |
| Compute | CPU-only temporal analysis |

## Direct dependencies

The pinned requirements are stored in
`tools/lf3r_annotator/requirements-analysis.txt`:

```text
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.9
```

Resolved uv pip freeze for the project interpreter:

    contourpy==1.3.2
    cycler==0.12.1
    fonttools==4.63.0
    kiwisolver==1.5.1
    matplotlib==3.10.9
    numpy==2.2.6
    packaging==26.3
    pandas==2.3.3
    pillow==12.3.0
    pyparsing==3.3.2
    python-dateutil==2.9.0.post0
    pytz==2026.3.post1
    six==1.17.0
    tzdata==2026.3

The environment also contains the compatible transitive packages installed by
uv. `uv pip check` completed successfully.

## Validation

```text
analysis dependencies OK
All installed packages are compatible
analyze_baseline_temporal_signals.py --help: OK
```

The repeatable setup command is:

```bash
bash tools/lf3r_annotator/setup_analysis_env.sh
```

The annotator reports the environment through `GET /api/health` and returns
`503` from `POST /api/analysis/run` if this interpreter or any required import
is unavailable. It never falls back to the server's system Python.
