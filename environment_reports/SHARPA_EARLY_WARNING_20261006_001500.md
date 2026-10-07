# Deform final-Align early warning

Completed 2026-10-06T13:14:22.604775+08:00; repository commit f46ce0111ed50472db2d4175786faf0ba8c66d22. Existing project-local ProcVLM/T-Rex Python, sequential GPU0 small frozen-feature heads. No raw annotation change, encoder finetuning/recompute, new environment/download or system change. GPU0 initially40330MiBfree; project disk56Gfree.

Exact commands:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.early_warning train --output outputs/sharpa_early_warning/20261006_001500 --device cuda:0 > logs/sharpa_early_warning_train_20261006_001500.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.early_warning report --output outputs/sharpa_early_warning/20261006_001500 > logs/sharpa_early_warning_report_20261006_001500_retry1.log 2>&1
```

H=0/8/15/30/45camera frames,30Hz;25newruns,seeds42–46. Only final Align failure end defines anchor. H>0 pre-anchor horizon positive and terminal ticks excluded from train loss,val checkpoint selection/main test;success allnegative,earlier Keys ignored.H0 terminalstate positive through observedend. FrozenDeform2560→Linear128→single causal GRU128→Linear1 sigmoid. Train-count balanced BCE weights N/(2Nc);natural val/test,threshold.5,train-only standardization,AdamW.001/.0001,batch8,max30/patience8,clip1. Original rollout split80/17/17;114rollouts,92success22failure.

PASS: source lastAlign event/key/outcome audit;all5H classcounts andterminal masks;25unique H/seed runs;bothval/test labels andframe/event confusion/BA/AP/ROC metrics replayed from saved scores;normalized matrices;weeklycopy SHA256 equality. Label-only audit outputs/sharpa_early_warning_labelaudit/20261006_125500. Minimal necessary checks;no reconstruction/extra model replay. Startup dominated by shared HDD I/O waits,not model training failures. Training exit0. First CPU report attempt hit a subplot index variable collision; corrected and report retry exited0,with no retraining or score changes.

All horizons reported; no test-selected horizon/threshold. Lead counted from firstvalid alarm,missing detections not zero-filled;positive-band event rate andcorrect-first-alarm rate distinguish premature alarms. PR-AUC uses AP,prevalence reported;balanced BCE score uncalibrated. Five seeds do not expand independent val/test3failure rollouts. No claim that annotation end independently proves physical irrecoverability.

[Original report](../outputs/sharpa_early_warning/20261006_001500/README.md),[Weekly report](../WeeklySummary/10.5/early_warning/20261006_001500/README.md).


## 执行结果解读

所有H均重复5seeds，未按test选择最佳H，也未调整阈值。下表强调首次报警是否落入对应目标时段；仅有较长lead或较高event recall不足以证明可靠预警。H0不是早期预测：其有效正标签在anchor及之后，首次过早报警为帧级FP。

| H帧 | Test BA % | Test PR-AUC(AP) % | Test frame FPR % | Event FPR % | Failure首报在正时段 % |
|---:|---:|---:|---:|---:|---:|
| 0 | 84.03 ± 11.61 | 11.40 ± 1.18 | 13.84 ± 2.41 | 57.14 ± 5.05 | 13.33 ± 18.26 |
| 8 | 68.98 ± 18.79 | 4.89 ± 4.11 | 23.70 ± 9.41 | 65.71 ± 17.79 | 0.00 ± 0.00 |
| 15 | 82.15 ± 3.41 | 8.72 ± 3.93 | 30.37 ± 2.04 | 68.57 ± 6.39 | 0.00 ± 0.00 |
| 30 | 80.61 ± 3.70 | 22.43 ± 13.60 | 31.90 ± 5.07 | 71.43 ± 11.29 | 6.67 ± 14.91 |
| 45 | 72.10 ± 11.13 | 21.83 ± 19.01 | 30.77 ± 5.25 | 72.86 ± 7.82 | 26.67 ± 14.91 |

**当前固定0.5阈值未形成可靠的within-H online预警。** 四个early-warning组的event FPR为65.71%–72.86%；failure首次报警落在对应正时段的比例按H=8/15/30/45依次为0.00%/0.00%/6.67%/26.67%。这些比例以全部failure rollout为分母，不仅是已报警样本。

较高frame BA/ROC-AUC及超过自然正例比例的AP提示Deform有可分信号，但大量首次报警出现在目标带之前；较长lead不能直接算可靠提前预警。本轮H0的持续状态检测BA较高，但四个early horizon的性能不单调，不能据此确定一个可靠预警horizon。上次F6单点Key任务与本轮的模态、目标和loss都不同，不作受控直接对比。

PR-AUC需对照上面的正例比例；val/test各只有3条最终failure，5seeds不增加独立failure样本数量。因此本轮可观察预警信号与误报的关系，尚不足以确认可部署的可靠预警horizon。原标注end作为用户定义的irrecoverable proxy，不代表新增物理不可恢复性判定。
