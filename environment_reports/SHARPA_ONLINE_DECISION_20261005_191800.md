# Online decision evaluation

Complete 2026-10-05T19:23:15.107573+08:00; commit f46ce0111ed50472db2d4175786faf0ba8c66d22. CPU probability postprocessing only, no inference/training/encoder changes. Fixedoriginal_align/DeformGRU/sigma8/seed42/checkpoint15, originalrolloutsplit. Step augmentation request superseded; audit only completed, no augmentation training started.

Command:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.online_decision_eval --output outputs/sharpa_online_decision/20261005_191800 > logs/sharpa_online_decision_20261005_191800.log 2>&1
```

441threshold pairs, validation-only first-trigger balanced event accuracy; undecided countedwrong. Selectedtau_decide=.55,tau_fail=.10. Test24interaction: ACC62.50, success accuracy60, failure accuracy66.67, BA63.33, macroF164.57, failureprecision50,recall66.67,FPR40,undecided8.33 (%). Firsttrigger median relativeKey−1.15seconds, correct-only−.8667seconds. 19/22returnprogress,9/22oppositeoutcome aftertrigger; firstlatched decision does not imply stable probabilities. Supplementary last-frame ACC33.33%,undecided33.33%; no first-trigger detection times attached to this secondary task.

PASS: thresholds searched exclusively onval22interactions; firsttrigger rules and confusion supports recomputed; originalsensor/cache/checkpoint/prediction hashes unchanged; all24testinteractions included andundecided retained; plots visuallyinspected, commonsecondaxis plusper-timeN. FourmeanPdecided/conditional panels cover allval/test success/failure interactions, conditional computed per tick before averaging. Source/copy hashes verified. [Fullreport](../outputs/sharpa_online_decision/20261005_191800/README.md), [Weeklyreport](../WeeklySummary/10.5/online_decision/20261005_191800/README.md).

CodeSHA256: 402390654a717ea6aa4dcdf3a26042f6da722e6ac3d01aeb2b1c4d702d38b7a7
