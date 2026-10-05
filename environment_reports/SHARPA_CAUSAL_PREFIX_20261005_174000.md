# Interval hindsight outcome / causal prefix — COMPLETE

Recorded: 2026-10-05T17:45:26.862944+08:00

Original annotations unchanged. Original259 event6/7/8/9 intervals, no Align merge or background. Rollout split80/17/17 unchanged; interval counts184/36/39 with success/failure129/55,28/8,29/10. One full dense valid-tick feature sequence and scalar outcome per interval, 16 sampled endpoint indices; no duplicated prefix files.

Prediction positions ceil(k*T/16)-1, k1..16, over valid sequence. Separate from local F6 encoder's past16-tick window. Both pretrained encoders frozen; interval-only verified featurecache reused. Deform at same currenttick. Causal GRU outputs eachtick; MLP uses cumulative past mean. Binary scalar logits→sigmoid. BCE pos_weight129/55; mean16slots per interval then meanintervals;184intervals/2944slots each epoch. Checkpoint selection validation mean16-position BA, fixed testthreshold.5.

Environment repos/ProcVLM/.venv, torch2.10.0+cu128, cuda:0; GPU jobs sequential. No environment/system changes or downloads. Disk free before run134GiB. Gitf46ce0111ed50472db2d4175786faf0ba8c66d22. Training duration282.3s;30runs, 6models×5seeds42–46.

Commands after source ./project_env.sh:
```bash
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.causal_prefix prepare --source outputs/sharpa_interval_binary/20261004_180632 --output outputs/sharpa_causal_prefix_data/20261005_174000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.causal_prefix train --source outputs/sharpa_causal_prefix_data/20261005_174000 --output outputs/sharpa_causal_prefix/20261005_174000 --device cuda:0
PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_causal_prefix --output outputs/sharpa_causal_prefix/20261005_174000
```

Verification PASS:30position metrics recomputed, allcheckpoint train-onlynormalization checked, originalsource/sensor/encoder hashes unchanged, split/closedbounds/endpoints/outcome labels verified. Six seed42 checkpoints CPUreplayed, futureperturbation unaffectedprefix, truncation equivalence, streaming equivalence, intervalreset passed. Streaming interface tools/sharpa_tactile/prefix_inference.py accepts currentencodedfeatures and does not require interval end time.

Best mean-prefix BA: Fusion MLP61.87±2.82%. Best100% BA: Fusion GRU70.83±5.58%, macroF166.36±4.59%. Prefix outcomes are correlated hindsight supervision, not16independent samples. Test39intervals come from17rollouts; event7 has only4intervals total(3train/0val/1test), insufficient evidence for Insert failure generalization. Fullinterval stage9 used merged114Align samples, so numerical change cannot isolate prefix-supervision effect.

Reports: outputs/sharpa_causal_prefix/20261005_174000/README.md; WeeklySummary/10.5/causal_prefix/README.md; dataset README under outputs/sharpa_causal_prefix_data/20261005_174000. Logs: logs/sharpa_causal_prefix_prepare_20261005_174000.log, logs/sharpa_causal_prefix_train_20261005_174000.log, logs/sharpa_causal_prefix_report_20261005_174000.log. Code snapshots in original outputs.
