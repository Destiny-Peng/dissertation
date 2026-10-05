# Merged Align interval binary separability — COMPLETE

Recorded: 2026-10-05T17:27:16.755841+08:00

Both current Key groups have identical merged intervals. Removing Key bands/background yields one unique corpus, verified via equal SHA256. Executed 6 models × seeds42–46 =30 unique runs, shared by symmetric_merged/asymmetric_merged.

114 independent intervals (one per rollout): success92/failure22. Split unchanged80/17/17, class counts64/16,14/3,14/3. Input is interval-only step3 sliding16 raw points→one F6 feature1280D plus last-point Deform2560D per window. All10628 windows reused from the corrected single-encoding cache; interval bounds verified. MLP averages full interval window features; causal GRU uses last hidden over full valid sequence. One binary prediction/loss per interval. Weighted CE by interval counts [.625,2.5]; val BA chooses checkpoint; test threshold=.5. No Key/in_progress labels, frame weighting, augmentation, encoder updates or test threshold tuning.

Environment repos/ProcVLM/.venv, torch 2.10.0+cu128, cuda:0; GPU stages sequential. Disk free before run138GiB. Git f46ce0111ed50472db2d4175786faf0ba8c66d22. Training duration49.8s. No installation/download/system changes or original data modifications.

Commands after source ./project_env.sh:
```bash
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_interval_binary prepare --source outputs/sharpa_merged_online_datasets/20261005_164000 --output outputs/sharpa_merged_interval_binary_data/20261005_172000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_interval_binary train --source outputs/sharpa_merged_interval_binary_data/20261005_172000 --output outputs/sharpa_merged_interval_binary/20261005_172000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_merged_interval_binary --output outputs/sharpa_merged_interval_binary/20261005_172000
```

Verification PASS: all30 metrics recomputed, split/bounds/GT/source hashes checked, train-only normalization checked against allcheckpoints, six checkpoint CPU replays; normalized confusion matrices checked. Sourcecode snapshots and hashes stored in original outputs.

Deform GRU: BA85.48±10.19%, macroF178.26±7.66%, AUROC89.52±2.13%; failure precision54.00%, recall86.67%. Deform MLP BA74.52±6.92%. F6 BA40–42% under these probes. Fusion fixed-threshold BA64–65%; GRU AUROC87.14%, but did not improve over Deform. Always-success ACC82.35% /BA50%; accuracy alone misleading. Test has only17 intervals /3failures; repeatseeds do not add independent testdata, so result is diagnostic evidence for thissplit, not a stable generalization claim.

Reports: outputs/sharpa_merged_interval_binary/20261005_172000/README.md; WeeklySummary/10.5/merged_interval_binary/README.md. Data docs: outputs/sharpa_merged_interval_binary_data/20261005_172000/README.md. Logs: logs/sharpa_merged_interval_binary_prepare_20261005_172000.log, logs/sharpa_merged_interval_binary_train_20261005_172000.log, logs/sharpa_merged_interval_binary_report_20261005_172000.log.
