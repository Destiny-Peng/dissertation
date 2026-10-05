# Key-aligned probabilities: all validation/test interactions

Completed 2026-10-05T18:50:57.947052+08:00; commit f46ce0111ed50472db2d4175786faf0ba8c66d22. Project-local T-Rex/ProcVLM Python, CPU inference only; no training, weights, labels, splits or selected sigma changed.

Command from project root:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.align_gaussian_probability_curves --source outputs/sharpa_gaussian_online/20261005_182000 --output outputs/sharpa_gaussian_online/20261005_182000/key_aligned_20261005_184000 > logs/sharpa_gaussian_key_aligned_20261005_184000.log 2>&1
```

PASS. Two groups, val/test separated,6model panels each,5seeds each selected configuration. All interactions: merged val14success/3failure,test14/3; originalAlign val14/8,test15/9. Four probability curves per panel grouped by final interaction outcome. Horizontal time=(last camera frame−annotation Key frame)/30s. Duplicate camera-frame ticks averaged before five-seed averaging; each interaction then one equal vote/time. Missing positions not filled; support N plotted. Variance ddof1 across seed-averaged interactions, undefined if N<2. Main shading mean±SD; literal mean±variance supplied. Seed mean variance retained separately. All22752CSV rows' means/variances recomputed against preserved per-interaction aligned arrays. Source and checkpoint hashes verified unchanged; both new and existing weekly copy manifests verified. All8PNG and8PDF plots present, sample image visually inspected.

Validation probabilities recomputed onCPU; test savedGPU probabilities reused unchanged. Max validation CPU vs savedGPU BA difference 0.0004823927 (no model selection rerun). Outputs retain predictions and aligned arrays for independent recomputation; weekly copy contains all plots/statistic tables/docs. [Report](../outputs/sharpa_gaussian_online/20261005_182000/key_aligned_20261005_184000/README.md), [weekly report](../WeeklySummary/10.5/gaussian_online/key_aligned_20261005_184000/README.md).
