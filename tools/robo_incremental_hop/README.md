# Robo-Dopamine four-signal hop failure analysis

This module performs CPU-only post-processing over already saved Robo-Dopamine
multi-perspective outputs and existing LF3R human annotations. It compares the
saved `hop` signal from:

- `incremental`
- `forward`
- `backward`
- `fused`

The same detector families, parameter grids, onset reference, clean false-positive
metrics, delay metrics, recovery diagnostics, and optional task-level CV are run
separately for each signal mode.

Event evaluation reports the complete native-sample delay profile
`Recall@1/@3/@5/@10/@20` plus `eventual recall`. Eventual recall means at
least one detector-positive native sample from `observable_onset_frame` until
the end of that failure episode. The earliest available boundary among
`recovery_frame`, `terminal_failure_frame`, and the next annotated observable
onset is treated as exclusive; if none exists, the final native sample in the
rollout is included. Detector grids and the existing best-config selection rule
remain unchanged; this is an evaluation expansion, not another threshold search.

For direct comparison, only rollouts with all four saved signals are included.
Native Robo-Dopamine sample frame indices are preserved and no interpolation is
performed.

`incremental.hop` is the official raw score and legacy percentage-point storage
is normalized only when confirmed. `forward.hop`, `backward.hop`, and
`fused.hop` are saved progress-difference signals and are used without
rescaling.

Run from the repository root:

```bash
source ./project_env.sh
python3 tools/analyze_robo_dopamine_incremental_hop.py \
  --run-root outputs/baselines/<completed-multi-perspective-robo-run>
```

Add `--task-cv` for optional leave-one-task-out parameter selection and held-out
evaluation. Add `--no-plots` to write only CSV/JSON artifacts. `--selection`
restricts a superset run to requested rollout IDs; the final analysis still uses
the four-signal intersection within that selection.

Required outputs are:

- `sweep_summary.csv`
- `event_results.csv`
- `clean_rollout_results.csv`
- `best_configs.csv`
- `recovery_results.csv`
- `metadata.json`

All result tables include `signal_mode`. Additional outputs are
`breakdown_summary.csv`, optional `task_cv_results.csv`, and mode-separated
plots under `plots/<signal_mode>/`.
