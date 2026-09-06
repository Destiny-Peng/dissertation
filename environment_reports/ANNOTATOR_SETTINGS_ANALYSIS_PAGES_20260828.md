# LF3R Annotator Settings and Analysis Pages

Date: 2026-08-28

## Result

The local annotator now has hash-routed `Review`, `Analysis`, and `Settings` pages in the existing native HTML/CSS/JavaScript application.

- `Review` keeps the existing video annotation workflow and per-rollout baseline cards.
- `Analysis` recalculates live manifest/annotation statistics and reads the newest complete baseline temporal-analysis snapshot without running inference.
- `Settings` previews and saves the shared appearance configuration at `config/lf3r_annotator.json`.

Semantic causal, observable, terminal, recovery, and controlled-data colors remain fixed. Font scaling uses the root `rem` scale; CSS `zoom` is not used.

## API and data behavior

`GET /api/settings` returns the project settings and defaults. `PUT /api/settings` accepts the exact fixed schema, validates six-digit hex colors, font scale `0.85`–`1.30`, and the three density values, then uses a temporary file and `os.replace` for persistence. Missing or malformed settings fall back to defaults.

`GET /api/analysis` selects the newest complete directory under `outputs/baseline_signal_analysis/`. It returns source metadata, selection count, manifest hashes, freshness status, all required summary tables, and compact event metrics joined to rollout/task metadata. Large frame arrays are excluded and clean-background rows are not sent as event rows.

The current checkout exposes `outputs/baseline_signal_analysis/full_136_20260827` with 125 selected rollouts, 432 compact event-signal rows, and coverage rows for SAFE, ProcVLM, RynnValue, and Robo-Dopamine. The snapshot is visibly stale because its recorded manifest hash/selection predates the current 136-row manifest; this is reported as provenance rather than hidden.

## Validation

- `/usr/bin/python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v`: **16 tests passed**.
- `/usr/bin/python3 -m py_compile tools/lf3r_annotator/server.py tools/lf3r_annotator/verify_pipeline.py tools/lf3r_annotator/tests/test_contract.py tools/lf3r_annotator/tests/test_server.py`: **passed**.
- GJS `new Function` parsing: `app.js` and `workspace.js` both reported **syntax ok**.
- Loopback smoke on a temporary port: `/api/health`, `/api/settings`, `/api/analysis`, `/`, and `/static/workspace.js` returned successfully; the real Analysis API returned the expected current snapshot metadata and 432 event rows.
- The Firefox headless attempt loaded the page and requested the new assets/APIs, but screenshot/DOM export was unavailable because the host software compositor failed to map its default framebuffer (`RenderCompositorSWGL failed mapping default framebuffer`). No GPU inference or baseline job was started by this work.

The detailed implementation and usage notes are in `tools/lf3r_annotator/README.md`.
