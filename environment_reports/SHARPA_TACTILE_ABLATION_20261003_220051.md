# Sharpa frozen tactile binary ablation — complete

Recorded: 2026-10-03T22:06:46.022619+08:00

Canonical report: [experiment README](../outputs/sharpa_tactile_ablation/20261003_220051/README.md).

Completed all six groups: F6 / Deform / F6+Deform × MLP / causal LSTM. Label 6/7 = failure, 8/9 = success; only annotated causal-to-observable intervals receive supervision. Frozen F6 and Deform encoders; user-approved 2×2 adaptive pooling yields 512D per finger. Dataset split is 80/17/17 whole USB rollouts, seed 42. Feature extraction and the six small probe training jobs ran sequentially on cuda:1. No other jobs were interrupted.

Environment: repos/ProcVLM/.venv, Python 3.10.21, torch 2.10.0+cu128. No dependencies were added for this experiment. T-Rex commit: f88e10c61da123c68bf0927cf4860bc97a0381f3; LF3R base commit: bd1696d81e0bd70a581767a70a037c1c47f38ecb. Local implementation file hashes are captured in experiment_manifest.json.

The README contains exact prepare, train, and verification commands. Final logs:
- logs/sharpa_tactile_prepare_20261003_220051.log
- logs/sharpa_tactile_train_20261003_220211.log

Validation: split disjointness, binary/conflict/background handling, six-group causality and streaming-state equivalence, frozen encoder parameters/state, and real raw-input online versus GPU cached-feature equivalence all PASS. Online maximum probability difference: 1.4901161193847656e-06. F6 reconstruction was checked once; Deform reconstruction omitted per user instruction. Original annotation backups remain under outputs/usb_event_intervals/20261003_202352/original_annotations_backup_20261003_203655; original files were not edited.

Old diagnostic output outputs/sharpa_tactile_ablation/20261003_215312 is marked SUPERSEDED. The final run disables TF32 and floors feature standard deviations at 0.01 to prevent near-constant dimensions amplifying floating-point noise. Test labels did not select these numerical settings.

Results are an initial single-seed, single-split comparison. Evaluation covers annotated intervals; background false alarms and inference latency were not benchmarked.
