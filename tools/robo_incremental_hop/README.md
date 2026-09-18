# Robo-Dopamine incremental-hop failure analysis

This module is CPU-only post-processing over already saved Robo-Dopamine
incremental `pred_vllm.json` outputs and the existing LF3R human annotations.

It intentionally does **not** use accumulated/fused progress, generic
change-point detection, whole-rollout Q95/std statistics, or model inference.

Run from the repository root:

```bash
source ./project_env.sh
python3 tools/analyze_robo_dopamine_incremental_hop.py \
  --run-root outputs/baselines/<completed-robo-dopamine-run>
```

Add `--task-cv` for optional leave-one-task-out parameter selection and
held-out evaluation. Add `--no-plots` to write only CSV/JSON artifacts.
`--selection <path>` restricts a superset completed run to the rollout IDs
listed in a project-local JSON selection document; selected IDs without a
completed Robo-Dopamine job are rejected rather than silently dropped. The
WebUI Analysis runner uses this option for its scope selector.

The analysis preserves Robo-Dopamine's native sampled frame indices. Current
official inference already stores incremental hop in `[-1, 1]`; the loader
cross-checks the saved `<score>...%</score>` text and only divides by 100 when
a saved result is confirmed to use percentage-point hop values.

Required outputs are:

- `sweep_summary.csv`
- `event_results.csv`
- `clean_rollout_results.csv`
- `best_configs.csv`
- `recovery_results.csv`
- `metadata.json`

Additional outputs are `breakdown_summary.csv`, optional
`task_cv_results.csv`, and plots under `plots/`.
