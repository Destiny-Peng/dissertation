# Event3-only frame experiment

COMPLETE at 2026-10-06T19:24:47.375679+08:00. HEAD `57358ff76818aa3388284f28349995b62ad2a706`.

15 frozen-feature GRU runs on cuda:0;same23rollouts19train4val,no test;class3 union inside1,allelse0;valBA epochselection,5seeds42–46. No encoder extraction/retraining. Validation metrics replay PASS;weekly copySHA PASS.

Environment: project-local ProcVLM venv via tools/run_trex.sh. Run prepare/train/report sequentially:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_three_only prepare --source outputs/sharpa_failure_relabel/20261006_165000 --output outputs/sharpa_failure_relabel/20261006_185200_frame3 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_three_only train --source outputs/sharpa_failure_relabel/20261006_165000 --output outputs/sharpa_failure_relabel/20261006_185200_frame3 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_three_only report --source outputs/sharpa_failure_relabel/20261006_165000 --output outputs/sharpa_failure_relabel/20261006_185200_frame3 --device cuda:0
```

Report: `WeeklySummary/10.5/failure_relabel/20261006_185200_frame3/README.md`.

Operational note: GPU training reached TRAIN_COMPLETE15 and saved all15 metrics/checkpoints/predictions/results.json. The worker remained in disk-I/O exit cleanup; after confirming15 completed records and15 checkpoints, SIGTERM was used to release that completed worker. The supervising shell returned1 due to termination; no training run was aborted. Report was then run separately with --training-shutdown terminated_after_complete; all saved validation metrics were replayed successfully.
