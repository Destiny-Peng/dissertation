# Early-warning branches A/B

Completed 2026-10-06T14:08:52.968510+08:00. Current repository commit57358ff76818aa3388284f28349995b62ad2a706. Existing project-local ProcVLM/T-Rex Python. User explicitly requested parallel branches: A50GPU0 frozen-feature heads sequential, BCPU75calibrations concurrent. GPU0 initial39986MiBfree,projectdisk52Gfree. No encoderfinetune/recomputation/download/systemchanges. Existing Deform25runs reused;rawannotations/Deformcheckpoints/probabilities unchanged.

Exact successful commands:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.early_warning_modalities train --output outputs/sharpa_early_warning_modalities/20261006_135300 --device cuda:0 > logs/sharpa_early_warning_modalities_train_20261006_135300.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.early_warning_modalities report --output outputs/sharpa_early_warning_modalities/20261006_135300 > logs/sharpa_early_warning_modalities_report_20261006_135300.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.calibrate_early_warning --output outputs/sharpa_early_warning_calibration/20261006_135300 > logs/sharpa_early_warning_calibration_20261006_135300.log 2>&1
```

A:H0/8/15/30/45×seeds42–46×F6/Fusion=50newruns;25Deformbaseline,reused.Total75verifiedresults.Same GT/terminal masks,classweights,normprotocol,rollout split,optimizer,valBA checkpoint selection. F6frozenfinger1280→Linear128→GRU128;Fusion2×Linear128→concat256→GRU128;sigmoidhead. Dense0,noaugmentation. Only train projections/GRU/head.

B:25existingDeformmodels×eventFPRbudgets.10/.15/.20=75thresholds. Eachval maxvalidrisk event score,maximumfailrecallunderFP budget,tiehighestthreshold,fixedtest. Float64 comparisons;noalert sentinel>1. FPR15/20identicalbecause14successallowFP2;10allowsFP1. No test-based threshold orH selection. Source AP/ROC unchanged.

PASS:minimal necessary label/count/weight equality checks;50newruns/75comparison Hseed coverage;allval/testframe/eventconfusion/BA/AP/ROC recomputed fromscores;75calibrationsvalFPconstraintsverified;allnormalized matrices;curveN/variance;source scoresSHA unchanged;weeklycopiesSHA match. Atrain/reportexit0. Binitial superseded outputs excluded;finalverificationPASS. No extra inference/reconstruction/test suites.

[Aoriginal](../outputs/sharpa_early_warning_modalities/20261006_135300/README.md),[Boriginal](../outputs/sharpa_early_warning_calibration/20261006_135300/README.md),[integratedsettingandresults](../WeeklySummary/10.5/10.5criticalreward.md).


## 后续并行实验：A模态对照、B事件阈值校准（2026-10-06）

A新增F6/Fusion各25runs，复用Deform25runs，共75模型结果。同H={0,8,15,30,45}、seeds42–46、rollout split、balanced BCE、GRU128及optimizer；F6冻结finger encoder1280→Linear128，Fusion两模态各Linear128→concat256→GRU128。仅最后Align failure end为anchor，label/mask与上一轮完全一致。

### A：Test PR-AUC（AP），5seed mean±SD

| H帧 | Deform AP % | F6 AP % | Fusion AP % |
|---:|---:|---:|---:|
| 0 | 11.40 ± 1.18 | 23.43 ± 12.61 | 11.43 ± 1.40 |
| 8 | 4.89 ± 4.11 | 2.29 ± 0.97 | 2.91 ± 0.80 |
| 15 | 8.72 ± 3.93 | 4.34 ± 1.74 | 5.16 ± 0.64 |
| 30 | 22.43 ± 13.60 | 7.87 ± 2.82 | 8.79 ± 4.16 |
| 45 | 21.83 ± 19.01 | 13.69 ± 6.80 | 12.91 ± 5.23 |

Deform early-warning平均AP最高，F6终态检测最强；Fusion无稳定提升。AP/ROC、同seed差值、完整GT/loss与曲线见[模态对照报告](early_warning_modalities/20261006_135300/README.md)。

### B：Deform event-level threshold calibration

不重训Deform，直接使用原checkpoint缓存的risk。每H×seed独立仅在val按有效范围max_t risk选择threshold：event FPR≤15%下最大failure recall，平分取更高threshold；固定test。Val14success/3failure：15%允许FP≤2，10%≤1，20%≤2，因此15%和20%阈值完全相同。阈值采用float64精确比较，必要时>1 sentinel表示关闭报警。

下表是主15% operating point的test均值±5seed SD；此val约束不保证每个test seed都满足15%。首报必须在对应positive band内才算有效预警，漏报不填0秒。

| H帧 | Event precision % | Recall % | FPR % | 正时段命中 % | 正确首报 % | 有效lead秒 | 有效seed数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 4.29 ± 3.91 | 0.00 ± 0.00 | 0.00 ± 0.00 | 无有效报警 | 0/5 |
| 8 | 13.33 ± 18.26 | 26.67 ± 36.51 | 12.86 ± 14.64 | 13.33 ± 18.26 | 0.00 ± 0.00 | 无有效报警 | 0/5 |
| 15 | 5.00 ± 11.18 | 6.67 ± 14.91 | 4.29 ± 9.58 | 0.00 ± 0.00 | 0.00 ± 0.00 | 无有效报警 | 0/5 |
| 30 | 18.00 ± 24.90 | 20.00 ± 29.81 | 5.71 ± 9.31 | 20.00 ± 29.81 | 13.33 ± 18.26 | 0.40 ± 0.14 | 2/5 |
| 45 | 10.00 ± 22.36 | 20.00 ± 44.72 | 4.29 ± 9.58 | 20.00 ± 44.72 | 6.67 ± 14.91 | 0.30 | 1/5 |

误报显著下降，但early-warning event recall仅6.67%–26.67%，正确首报覆盖很低：H30只有2/5 seeds、H45只有1/5出现有效首报。故不能认为只是0.5阈值不合适，当前checkpoint在此val-only calibration下仍没有形成可靠alert。原frame/eventAP和ROC完全不变。H0不称early warning。

[阈值、10%/20% sensitivity、校准前后比较和margin曲线](early_warning_calibration/20261006_135300/README.md)。

两条线GPU小模型训练与CPU calibration独立执行；没有根据test选模型、H或threshold，没有改原始标注、原Deform checkpoint或risk缓存。所有验证及复制hash通过。
