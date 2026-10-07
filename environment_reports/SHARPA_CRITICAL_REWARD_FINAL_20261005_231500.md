# Final Align failure Key critical reward sweep

Completed 2026-10-05T23:43:22.816238+08:00; repository commit f46ce0111ed50472db2d4175786faf0ba8c66d22. Existing project-local ProcVLM/T-Rex Python; GPU0 sequential small frozen-feature heads only. No raw annotation changes, encoder training/recomputation, downloads, or system changes.

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.critical_reward_final_fast train --rule last --grid coarse --output outputs/sharpa_critical_reward_final/20261005_231500 --device cuda:0 > logs/sharpa_critical_reward_final_train_20261005_231500.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.report_critical_reward_final --output outputs/sharpa_critical_reward_final/20261005_231500 > logs/sharpa_critical_reward_final_report_20261005_231500.log 2>&1
```

35 configurations (n=0,5,10,15,20,25,30; m=0,3,5,10,15) × independent seeds42–46 =175new runs. Last Align only; finalfailure band positive, finalsuccess and earlier Keys ignored. Earlier sensor history retained. Original annotation observation range and rollout split unchanged: train80(64success/16failure), val17(14/3), test17(14/3). Dense F6 cached finger-mode1280D→Linear128→causal GRU128→Linear1 sigmoid; plain BCE, threshold0.5. Best epoch by own-band val BA; n/m by mean5seed val BA, never test. Each band changes frame GT; fixed rollout event GT reported separately.

PASS: final-Key/outcome label audit against last source Align event; all35 train/val/test class distributions;175result cardinality and unique seed/config combinations; all175test labels, confusion matrices, BA and MacroF1 independently recomputed; weekly copies SHA256 verified. Key-relative curve means average seeds inside rollout, then available rollouts; CSV includes variance and N. Selected confusion CSV includes mean counts and row normalization. Minimal checks only; no decoder/reconstruction/extra retraining.

Selected n0,m0: valBA62.90±11.76%, testKeyBA45.47±4.90%, MacroF147.48±2.75%, precision/recall0%, FPR9.05±9.81%. Fixed testeventBA37.38±6.87%, recall13.33±18.26%, FPR38.57±28.84%. All175test Key recalls0. Result does not establish reliable detector or encoder nonseparability. Only3failure rollouts per val/test and rare Key positives; limitations recorded.

Startup delayed by shared-HDD I/O waiting; removed unused Deform/HuggingFace imports without changing architecture. Earlier drafts/logs were stopped before training and excluded; no obsolete result reused. Label-only audit: outputs/sharpa_critical_reward_final_labelaudit/20261005_231000.

All175 results and TRAIN_COMPLETE were saved before shutdown. The training process subsequently stalled during exit in HDD I/O wait; after independent result verification and report completion it received SIGTERM to release resources (exit143). This was post-training cleanup, with no interrupted run or missing artifact. Report exited0; heatmap colorbar placement corrected without inference or metric changes.

[Original report](../outputs/sharpa_critical_reward_final/20261005_231500/README.md), [Weekly report](../WeeklySummary/10.5/critical_reward_final/20261005_231500/README.md).
