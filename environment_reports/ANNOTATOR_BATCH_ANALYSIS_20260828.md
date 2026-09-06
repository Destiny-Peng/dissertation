# LF3R Annotator Batch Baseline and Temporal Analysis Report — 2026-08-28

## Result

The annotator now fixes baseline chart/timeline alignment and can launch one-method baseline batches from Review or run the existing temporal analyzer from Analysis. Baseline methods use independent persistent tmux jobs and may run concurrently; rollout generation remains the exclusive manifest-writing task.

## Implementation

- Baseline SVG curves and onset markers use a normalized `0 ... 100` x-domain, clamp to the video frame domain, and share `.timeline-track` geometry with the playback slider. The negative horizontal chart margin was removed.
- Review exposes batch scope selection for `all`, `natural_observation`, `primary_natural`, `reference_natural`, and `controlled_analysis`. Primary/reference scopes map to `--partition natural_observation --dataset-role ...`; the other scopes map directly to the runner partition.
- Batch requests accept `0` or comma-separated GPU IDs, a free-memory fraction, zero-based start index, optional limit, and method-specific runner options. The memory value is interpreted relative to currently free GPU memory; GPU utilization is not used as a readiness requirement.
- `GET /api/baselines/runs` discovers complete project-local runs and reports rollout-ID compatibility. `POST /api/baselines/run-batch` writes under `outputs/baselines/web_runs/` and exposes progress/log endpoints.
- Analysis exposes four compatible run selectors. `POST /api/analysis/run` creates a hidden selection file, verifies that each selected run covers every requested rollout ID (superset runs are valid), runs the existing analyzer without model inference, and atomically moves a complete snapshot into `outputs/baseline_signal_analysis/`.
- Analysis job status/log endpoints refresh the snapshot automatically after completion. Missing raw output remains method coverage information instead of preventing a valid partial analysis.
- Review, Analysis, and Settings use flexible zero-minimum grids and wrapping controls. Root rem font scaling remains 75%-160%, with independent Review, Analysis, and control/help scales at 85%-130%; Compact/Comfortable/Spacious density remains supported.

## Verification

- `python3 -m unittest discover -s tools/lf3r_annotator/tests -p 'test_*.py' -v`: **28 tests passed**.
- python3 -m py_compile tools/lf3r_annotator/server.py tools/lf3r_annotator/task_supervisor.py tools/lf3r_annotator/verify_pipeline.py tools/analyze_baseline_temporal_signals.py: passed.
- GJS `new Function` parsing: `app.js` and `workspace.js` both reported **syntax ok**.
- HTML parser smoke: `index.html` parsed with 121 unique IDs and no duplicate IDs.
- CSS guard: no legacy negative horizontal timeline margin or fixed pixel font-size regression remains.
- Read-only real-output discovery: 19 completed run roots were found; `primary_natural` had compatible SAFE/ProcVLM/RynnValue/Robo-Dopamine candidates, while the empty `controlled_analysis` scope correctly reported zero compatible runs.
- The unit tests exercised baseline batch command mapping, free-memory validation, path/scope checks, shared-job `409` conflicts, job progress/log endpoints, superset run-ID validation, atomic temporal-analysis output, and unavailable cases.
- No baseline GPU inference or temporal-analysis job was started during this implementation verification.

## Files

See `tools/lf3r_annotator/README.md` for UI/API usage and runner parameter mapping. The existing `full_136_20260827` snapshot remains available through `/api/analysis`; new web analyses are clearly marked by their `web_<timestamp>_<label>_<job-suffix>` output directory.
