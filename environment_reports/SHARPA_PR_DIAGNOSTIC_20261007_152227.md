# Validation-only PR-AUC early-warning diagnostic

Completed: 2026-10-07T15:28:38.705087+08:00

User authorized H0/8/15, seed42, max30/patience8, PR-AUC selection, every epoch state_dict, no test evaluation/calibration. Original Deform GRU balanced-BCE training unchanged. GPU parallel0:H0,1:H8,2:H15; explicit user GPU parallel authorization overrides repository default sequential guideline. Available disk168GB before run; epoch weights123.9MiB,75 files including three initial states. Uses project-local repos/ProcVLM/.venv; no install/download.

```bash
source ./project_env.sh
bash tools/run_early_warning_pr_diagnostic.sh outputs/sharpa_early_warning_pr_diagnostic/20261007_152227
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -u -m sharpa_tactile.early_warning_pr_report --output outputs/sharpa_early_warning_pr_diagnostic/20261007_152227
```

H0 best14 stopped22; H8/H15 best17 stopped25. All epoch raw state_dicts saved with train normalization buffers, checkpoint index SHA and best file SHA match. All val score metrics replayed. Old train-BCE/BA history overlap reproduces exactly. Positive-ranking metrics use logits to avoid sigmoid saturation ties. CSV includes natural and train-weighted validation BCE, neither used for epoch selection. test not inferred/scored and no threshold search.

Previous log-only BA300 run outputs/sharpa_early_warning_epoch_metrics/20261007_151852 had already completed when user changed protocol; preserved separately. It is not this PR-selected diagnostic and its test metrics were not used here. New run only H0/8/15.

诊断结论：H8/H15从epoch2到epoch17，BA@0.5降至约50%，但AP分别从0.0344→0.2467、0.0556→0.1472，ROC-AUC也提升，支持这一阶段存在明显的score尺度/固定阈值问题。最佳AP时两组正例score最大值仅0.4293/0.3172，因此0.5阈值没有检出正例。继续到epoch25，train BCE继续下降，而val natural BCE上升、AP与ROC-AUC下降，说明后续也出现真实的validation泛化退化；两种现象可以先后存在。H0的AP提升同时ROC-AUC略降，表现不是完全一致的排序改善。当前仅validation诊断，不做阈值搜索或test验证。

Independent raw-state_dict CPU inference on validation H8 first32 ticks succeeded. CPU/CUDA maxlogit difference1.836e-4, maxprobability difference8.032e-6; initial1e-4 absolute logit assertion too strict, recorded exact observed differences rather than claiming bitwise equality. Full checkpoint SHA and all saved val-metric replay checks passed.
