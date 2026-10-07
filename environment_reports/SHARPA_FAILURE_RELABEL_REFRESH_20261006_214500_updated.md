# Refreshed3/4 frame training and success frame3 audit

COMPLETE:2026-10-06T22:34:02.843618+08:00. HEAD:`57358ff76818aa3388284f28349995b62ad2a706`.

30frozen-feature GRU runs,24failure rollouts20train4val,no test.15frame3 checkpoints on92success;fixed.5 any detection=rolloutFP. Success never used in selection/normalization/loss. Project-local ProcVLM venv;GPUcuda:1;GPU work sequential in one process.

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_refresh run --source outputs/sharpa_failure_relabel/20261006_165000 --cache outputs/sharpa_tactile_three_class/20261004_160624 --frame3-baseline outputs/sharpa_failure_relabel/20261006_185200_frame3 --output outputs/sharpa_failure_relabel/20261006_214500_updated --device cuda:1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_refresh report --source outputs/sharpa_failure_relabel/20261006_165000 --cache outputs/sharpa_tactile_three_class/20261004_160624 --frame3-baseline outputs/sharpa_failure_relabel/20261006_185200_frame3 --output outputs/sharpa_failure_relabel/20261006_214500_updated --device cuda:1
```

CLI run exits only after all predictions/checkpoints/results and gpu_complete.json are saved and CUDA synchronized;os._exit0 bypasses slow unrelated Torch exit hooks.

Metrics replay andweeklycopySHA PASS. Report:`WeeklySummary/10.5/failure_relabel/20261006_214500_updated/README.md`.
