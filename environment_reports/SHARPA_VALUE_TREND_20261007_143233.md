# Tactile value-trend training record

Completed: 2026-10-07T14:42:39.319885+08:00

User authorized seed42 only for each H, shared proxy/future training, original historical final Align Key, and Key-H focal trigger + continuous K8 trend risk. User explicitly requested GPU parallelism; this overrides the repository's default sequential GPU rule for this experiment.

Commands:
```bash
source ./project_env.sh
export PYTHONPATH="$PROJECT_ROOT/tools"
"$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -u -m sharpa_tactile.value_trend prepare --output outputs/sharpa_value_trend/20261007_143233 --device cuda:2
bash tools/run_value_trend_parallel.sh outputs/sharpa_value_trend/20261007_143233
"$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -u -m sharpa_tactile.value_trend_report --output outputs/sharpa_value_trend/20261007_143233
```

Shared stages on GPU2, then H0/H30 on GPU0, H8/H45 on GPU1, H15 on GPU2 concurrently. Environment: project-local repos/ProcVLM/.venv. No installation/download or annotation mutation.

Output: outputs/sharpa_value_trend/20261007_143233
Weekly report: WeeklySummary/10.5/10.5value_trend.md

All7 best checkpoints satisfy lowest eligible validation loss; stopped exactly50 epochs after best, max300. Causal prefix inference check max error0 for proxy, 7.45e-7 for future. 114 cached interaction features generated from frozen models; split80/17/17. Normalization and all model parameters fitted from train only; val selects stopping and optional thresholds; test evaluated only after training.

Implementation notes: first warmup launch was interrupted before usable checkpoint and retained as *_warmup_interrupted; zero-gradient failure samples during success-only warmup were corrected before completed training. Calibration evaluation uses float64 to preserve the all-negative threshold just above maximum score; final events and metrics replayed from saved original probabilities without retraining. Report is a tactile-only observation-conditioned adaptation, not a UniIntervene reproduction (no recorded policy action, no language/V-JEPA2).
