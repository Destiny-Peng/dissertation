# Sharpa corrected online-window training — COMPLETE

Recorded: 2026-10-05T15:59:42.098191+08:00

Environment: repos/ProcVLM/.venv (PyTorch 2.10.0+cu128); GPU 0. GPU training was sequential; frozen cached encoder features only. Available disk before run: 162 GiB. Git: f46ce0111ed50472db2d4175786faf0ba8c66d22; existing user changes preserved.

Command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_sharpa_tactile_ablation.sh train_align_online --dataset outputs/sharpa_align_online_datasets/20261005_152000 --output outputs/sharpa_align_online_training/20261005_155000 --device cuda:0
bash tools/run_sharpa_tactile_ablation.sh report_align_online --output outputs/sharpa_align_online_training/20261005_155000
```

60 runs (two label groups, six models, seeds 42–46), max 30 epochs, patience 8. Training time: 324.9 seconds. Both ready training sets balanced; inverse-frequency CE weights=[1,1,1]. Fixed ready labels, rollout split, natural val/test. No encoder updates, downloads, reconstruction or original data edits.

Verification: PASS. Metrics and fixed arrays checked for all 60 runs; 12 seed-42 checkpoints replayed on CPU, six GRU causal checks passed. Data/encoder/code hashes unchanged. Exact source hashes/configuration: outputs/sharpa_align_online_training/20261005_155000/run_manifest.json.

Results: outputs/sharpa_align_online_training/20261005_155000/README.md. Weekly copy: WeeklySummary/10.5/align_online_training/README.md. Logs: logs/sharpa_align_online_training_20261005_155000.log and logs/sharpa_align_online_report_20261005_155000.log.

Baseline Deform GRU: test BA 49.54 ± 3.90%, macro F1 47.13 ± 3.00%. Asymmetric merged F6 MLP: test BA 57.61 ± 6.77%, macro F1 40.81 ± 3.36%; failure precision 10.90%, recall 78.00%. Fusion did not have the highest average BA within either group. Different group GTs prevent a direct performance-improvement claim; merged test failure has only three rollouts/60 overlapping windows.
