# Frozen tactile interval binary classification — complete

Recorded: 2026-10-04T18:13:48.953071+08:00. [Full report](../outputs/sharpa_interval_binary/20261004_180632/README.md), [five-seed comparison](../outputs/sharpa_interval_binary/20261004_180632/comparison.csv).

Completed 259 independent labeled interval samples, six model groups, five seeds 42–46 (30 small probe trainings sequentially on cuda:1). Label event 6/7 failure=1, event 8/9 success=0; background not sampled or supervised. Original causal/observable fields are used only as closed interval boundaries.

Rollout split byte-identical to source outputs/sharpa_tactile_three_class/20261004_160624; split SHA256 056aafb5bdb84300cdac38c36913cb9045be1f1c09f989fea674e1d43cf2f0e7. Train/val/test 80/17/17 rollouts, 184/36/39 intervals. Train class counts 129 success/55 failure; weights 0.7131782945736435 / 1.6727272727272726 computed from interval counts. Test supports 29 success/10 failure.

F6 strict interval-only sliding 16 ticks with first valid tick repeated for left padding; no pre-interval input. Deform timestep encoder unchanged, five-finger pooled 512D/finger. Both pretrained encoders frozen. MLP mean-pools timestep projected features before classification; LSTM classifies final valid hidden state and resets per interval. Fusion is timestep concat. One label/output/loss per interval, independent of duration; no frame-wise probability averaging. Train-only feature normalization uses interval data.

Optimizer and budget unchanged: hidden 128/LSTM one layer, AdamW .001/.0001, batch 8 intervals, max30/patience8, clip1, FP32/TF32 disabled. Best epoch validation interval BA at threshold .5. No test tuning.

Verification PASS: interval boundaries/labels, split/rollout isolation, interval-count class weights, temporal mean permutation invariance, padding isolation, packed final LSTM state/streaming equivalence, independent interval states, interval-unit metrics, immutable pretrained weights. One real F6 prefix CPU encoding vs GPU cache PASS (max abs feature diff 1.430511474609375e-06). No new reconstruction. Old frame-wise API rejects interval checkpoints; use IntervalProbe.

Environment repos/ProcVLM/.venv; Python3.10.21/torch2.10.0+cu128. No new dependency, no original annotation or old-result edits, no unrelated-process interruption. LF3R base commit f46ce0111ed50472db2d4175786faf0ba8c66d22; T-Rex commit f88e10c61da123c68bf0927cf4860bc97a0381f3. Code hashes in experiment_manifest.json.

Exact commands in canonical README. Logs logs/sharpa_interval_prepare_20261004_180632.log, logs/sharpa_interval_verify_20261004_180632.log, logs/sharpa_interval_train_20261004_180632.log, logs/sharpa_interval_report_20261004_180632.log. Extra real-prefix verification is integrated into verify_intervals and recorded in raw_interval_prefix_check.json.

Interval results cannot be compared directly with frame-wise or online onset localization metrics. Only 39 held-out intervals/10 failures; one fixed split and five training seeds. Pair differences recorded in paired_differences.json.
