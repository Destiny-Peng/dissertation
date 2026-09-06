# LF3R Annotator rollout generation

- Date: 2026-08-28
- Scope: OpenVLA + LIBERO-10 natural only
- Execution: browser job API over the existing generator wrapper, Python evaluator, and manifest builder
- Output confinement: outputs/openvla_libero/ and project-local logs/rollouts/web_runs/
- Job coordination: shared single-task lock with baseline and temporal-analysis jobs
- GPU policy: memory-only gate; at least 30 GiB free and less than 50% used; utilization is recorded but never used as a blocker
- Provenance: server-generated lf3r-data-natural-libero10- timestamped run note, optional sanitized label, overwrite protection
- UI: Review form exposes GPU, inclusive task range, trials, seed, and label; status and recent log are polled from the browser
- Validation: fake generator tests cover command argument mapping, requested/completed counts, manifest rebuild detection, log endpoint, memory_blocked, invalid input, and shared-job conflict
- Safety: no real GPU inference was started during this implementation; existing rollout, media, and annotation files were not modified
- Compatibility: temporal analyzer accepts the web-facing `--robo-dopamine-run` spelling and its historical underscore form.
