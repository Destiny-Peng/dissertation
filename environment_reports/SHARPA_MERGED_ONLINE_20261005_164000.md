# Sharpa merged Align / corrected window input — COMPLETE

Recorded: 2026-10-05T16:54:47.557929+08:00

114 rollouts: 92 success / 22 failure; 11 without target annotation excluded. Original rollout split remains 80/17/17 (success/failure: 64/16, 14/3, 14/3). Each rollout now has one merged Align interval, including gaps, and only its last Key. Two comparisons: symmetric_merged / asymmetric_merged.

Input corrected: exact16 raw F6 sample points→one frozen encoder output1280D (official finger-mode checkpoint); only last-frame deform→2560D. MLP classifies one window vector; causal GRU carries history between chronological windows, with reset at rollout/tactile gaps. Training sampling uses loss masks on full GRU sequences. Natural val/test retained, training classes balanced, weighted CE=[1,1,1].

Environment: repos/ProcVLM/.venv, PyTorch 2.10.0+cu128; device cuda:0; sequential GPU jobs. No downloads, environment changes, encoder updates, or original data changes. Available disk before run: 139 GiB. Commit: f46ce0111ed50472db2d4175786faf0ba8c66d22. Training time: 173.2s; 60 runs, seeds42–46, two groups×six models.

Commands (from PROJECT_ROOT after source ./project_env.sh):
```bash
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_online prepare --source outputs/sharpa_align_online_datasets/20261005_152000 --output outputs/sharpa_merged_online_datasets/20261005_164000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_online train --source outputs/sharpa_merged_online_datasets/20261005_164000 --output outputs/sharpa_merged_online_training/20261005_164000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_merged_online --output outputs/sharpa_merged_online_training/20261005_164000
```

Generation code pinned in dataset code_snapshot/; trainer and reporter pinned in training code_snapshot/. Dataset input_hashes point to immutable generation code snapshot; current CLI removed the unused report branch after generation, with no generation/training algorithm change. Separate reporter provides the report command.

Verification PASS: all60 metrics/GT arrays checked, original source-sensor hashes checked against previous audit, input hashes unchanged, 12 checkpoint CPU replays, six GRU causal and streaming-state equivalence checks. All60 normalized confusion matrices checked, seed42 count and row-normalized plots provided.

Best average BA in both groups: Deform GRU. Symmetric64.84±3.47%; asymmetric64.97±5.39%. Failure precision14.11% /17.45%, recall94.09% /86.67%; false positives remain. Only3 failure test rollouts. Changed GT and input/history prevent attributing score differences to merging alone.

Reports: outputs/sharpa_merged_online_training/20261005_164000/README.md and WeeklySummary/10.5/merged_online/README.md. Dataset report: outputs/sharpa_merged_online_datasets/20261005_164000/README.md. Logs: logs/sharpa_merged_online_prepare_20261005_164000.log, logs/sharpa_merged_online_train_20261005_164000.log, logs/sharpa_merged_online_report_20261005_164000.log.
