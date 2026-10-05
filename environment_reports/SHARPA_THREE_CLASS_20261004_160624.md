# Sharpa tactile three-class experiment — complete

Recorded: 2026-10-04T16:15:34.967932+08:00

[Canonical full report](../outputs/sharpa_tactile_three_class/20261004_160624/README.md) contains exact commands, six-group metrics, 3×3 confusion matrices, per-class precision/recall/F1, plots, and online API.

Hard frame labels: background=0, success(event 8/9)=1, failure(event 6/7)=2. Closed causal-to-observable intervals; no Gaussian, soft labels, smoothing, or -1 ignore target. Complete usable rollout suffixes now supervised as background. Video-frame label arrays cover every original frame; unchanged sensor validity filters and 16-tick warmup define valid model inputs. Loss excludes batch padding using lengths, rather than an ignore label.

Exactly reused the original binary split file, SHA256 056aafb5bdb84300cdac38c36913cb9045be1f1c09f989fea674e1d43cf2f0e7; train/val/test rollouts 80/17/17. Valid sample counts 47,915 / 9,794 / 10,207. Frozen encoders, pooling, projections, concat, model sizes, optimizer and training budget unchanged. Six small probe jobs ran sequentially on cuda:1; no other processes were interrupted and no dependencies were installed. Environment repos/ProcVLM/.venv: Python 3.10.21, torch 2.10.0+cu128.

Weighted CE uses N/(3*n_class), train-only weights [0.4295537267136428, 2.5295639320029566, 3.6143169646224638]. Selection by validation 3-class macro recall, argmax softmax inference. All label, split, padding, metric, causality, frozen-encoder and old binary-checkpoint compatibility checks PASS. Raw-input online three-class probabilities agree with GPU cached-feature output within 2.4437904357910156e-06. Existing F6 reconstruction reused, no new reconstruction.

T-Rex commit f88e10c61da123c68bf0927cf4860bc97a0381f3; LF3R base commit f46ce0111ed50472db2d4175786faf0ba8c66d22. Local source hashes saved in experiment_manifest.json. Logs: logs/sharpa_three_prepare_20261004_160624.log, logs/sharpa_three_verify_20261004_160624.log, logs/sharpa_three_train_20261004_160624.log, logs/sharpa_three_online_20261004_160624.log, logs/sharpa_three_report_20261004_160624.log.

Original annotations and old binary results preserved. Single-seed/single-split preliminary comparison. New background-inclusive 3-class results have a different evaluation target from old interval-only binary results.
