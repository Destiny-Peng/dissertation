# Sharpa class-weight and seed ablation — complete

Recorded 2026-10-04T17:42:38.928807+08:00. [Full experiment report](../outputs/sharpa_weight_seed_ablation/20261004_172727/README.md), [findings](../outputs/sharpa_weight_seed_ablation/20261004_172727/FINDINGS.md), [summary CSV](../outputs/sharpa_weight_seed_ablation/20261004_172727/summary.csv).

Completed 3 loss modes × 6 frozen probes × 5 training seeds (42–46): 90 results, 6 existing baseline groups imported and 84 small probes trained sequentially on cuda:1. No encoder training, data re-extraction, architecture edits, new dependencies, or other-process interruption.

Exact command: `bash tools/run_sharpa_tactile_ablation.sh weight_seed_ablation --source outputs/sharpa_tactile_three_class/20261004_160624 --output outputs/sharpa_weight_seed_ablation/20261004_172727 --device cuda:1`.

Environment repos/ProcVLM/.venv, Python 3.10.21, torch 2.10.0+cu128. LF3R base commit f46ce0111ed50472db2d4175786faf0ba8c66d22; T-Rex f88e10c61da123c68bf0927cf4860bc97a0381f3. Code hashes in suite_config.json. Log logs/sharpa_weight_seeds_20261004_172727.log.

Fixed labels, feature cache and original 80/17/17 rollout split, split SHA256 056aafb5bdb84300cdac38c36913cb9045be1f1c09f989fea674e1d43cf2f0e7. Hard background/success/failure labels unchanged. Inverse-frequency N/(3*n_class), unweighted all 1, inverse-sqrt normalized to arithmetic mean 1. Existing optimizer/model-size/batch/30-epoch/patience-8 settings unchanged. Validation macro recall selects checkpoints; test not used to choose weights or architecture.

Verification PASS: source labels/causality/padding contracts, immutable encoder identities, weight formulas and weighted CE scale invariance, independent training seeds/fixed split, 90 unique completed checkpoint groups, 18 complete five-seed cohorts, per-run class counts/weights and full test support. Source split/data manifest and encoder hashes unchanged after all jobs.

## Interpretation after all five seeds

以 Fusion LSTM 为例，inverse-frequency / unweighted / sqrt-inverse 的五 seed 平均 BA 为 63.40% / 55.77% / 58.56%，failure FPR 为 15.97% / 5.74% / 10.93%，failure recall 为 51.54% / 19.00% / 36.67%。弱化权重降低了误报，也增加了漏检；sqrt 的表现居中。failure precision 仅由 15.84% 变为 16.54% / 16.39%，不能把低 FPR 单独解释为可分性明显改善。Macro F1 分别为 56.92% / 57.67% / 56.66%，也说明 loss 选择取决于具体目标。

Inverse-frequency 下 Fusion LSTM 的 BA 均值±样本 SD 为 63.40%±2.03%，F6 LSTM 为 56.54%±4.04%，Deform MLP 为 58.68%±2.17%，Deform LSTM 为 61.31%±4.68%。Fusion 相对 F6 LSTM / Deform MLP 在五 seeds 均获胜；但对同 head 的 Deform LSTM 只在 3/5 seeds 获胜，平均差 +2.08pp，配对 seed 95% t 区间为 [-3.94,+8.10]pp，包含零。

换成 unweighted 或 sqrt 后，Fusion LSTM 与 Deform LSTM 的平均 BA 差分别为 -0.21pp 和 -1.53pp（各仅 2/5 seeds 获胜）；没有跨 loss 稳定的 Fusion 优势。互补性有更多证据，但仍未证明 Fusion 稳定优于最强单模态 LSTM。不能笼统从单次结果推断 F6 的 seed 稳定性：BA 的波动还依赖 head 与 loss。

这些结论只涵盖一个固定 rollout split 的训练随机性；未进行跨 split 或跨任务泛化验证。当前不继续改 architecture。

