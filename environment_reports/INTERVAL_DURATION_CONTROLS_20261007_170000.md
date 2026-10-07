# Duration controls / prefix experiment

Status COMPLETE: 60 new Deform+GRU runs +5 reused original checkpoints. GPU0/1/2 parallel jobs all exited0. Original dataset/split/annotation snapshots unchanged; no encoder training, no test, no F6/Fusion.

Git HEAD:57358ff76818aa3388284f28349995b62ad2a706. Project ProcVLM venv Python3.10. No dependencies installed. GPU free memory before launch approx17/24/19GB. Filesystem free175GB.

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.interval_duration_controls prepare --output outputs/sharpa_interval_duration_controls/20261007_170000
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.interval_duration_controls train --output outputs/sharpa_interval_duration_controls/20261007_170000 --device cuda:0 --variants length_matched prefix25 prefix50 prefix75
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.interval_duration_controls train --output outputs/sharpa_interval_duration_controls/20261007_170000 --device cuda:1 --variants normalized_K16 prefix25_K16 prefix50_K16 prefix75_K16
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.interval_duration_controls train --output outputs/sharpa_interval_duration_controls/20261007_170000 --device cuda:2 --variants normalized_K32 prefix25_K32 prefix50_K32 prefix75_K32
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.interval_duration_controls report --output outputs/sharpa_interval_duration_controls/20261007_170000
```

Train commands launched concurrently under prior user parallel-GPU authorization. prepare and report run sequentially. Verification: 65 probability/metric replays; source backup SHA equality; best-BA/earlystop30/patience8; rollout separation; weekly artifact/figure byte equality. Prefix chart visually inspected. No additional model smoke runs or training reruns. Full logs logs/interval_duration_*_20261007_170000.log.

Results and interpretation: WeeklySummary/10.5/10.5interval_duration_controls.md; raw output outputs/sharpa_interval_duration_controls/20261007_170000.
