# Early-warning dynamic threshold

Started 2026-10-06T23:13:25.502076+08:00; HEAD `57358ff76818aa3388284f28349995b62ad2a706`.

Environment: project-local ProcVLM venv via tools/run_trex.sh. GPU cuda:1; frozen risk curves, small MLP16/GRU16 threshold heads only. Source early-warning25 Deform checkpoints are unchanged. 3-fold validation-rollout OOF,50 final heads+150 fold fits; test excluded from fit/selection. No installations/downloads. GPU free32159MiB at startup, existing utilization100%; sequential small-head training. Disk25G free.

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.dynamic_threshold run --output outputs/sharpa_early_warning_dynamic_threshold/20261006_224500 --device cuda:1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.dynamic_threshold report --output outputs/sharpa_early_warning_dynamic_threshold/20261006_224500 --device cuda:1
```

Source AST check PASS. Completion/metrics replay will be recorded after runs.

Dynamic-threshold COMPLETE 2026-10-06T23:39:01.644772+08:00: frozen Deform early-warning curves;50MLP/GRU heads+150OOF fold fits;5H×5seeds. Original17val3foldcalibration,17test evaluation. Source syntax/prediction replay PASS;25 fixed thresholds and confusion matrices exactly match previous15% calibration;weeklycopySHA PASS. No stable advantage overfixedthreshold. Report:WeeklySummary/10.5/10.5early_warning_dynamic_threshold.md.

Dynamic threshold variation audit 2026-10-07T13:32:55.046921+08:00: project ProcVLM venv, CPU numpy/matplotlib, no retraining/checkpoint forward.850test sequences nonconstant;logit archive algebra maxerror5.63e-7. H45 medianrange MLP.00902/GRU.03017;preanchor excludingfirst16ticks .00298/.00816.20optimizerupdates perfit not verified converged;claims limited to nearconstant learned thresholds. Individual-seed change plot/CSV andREADME audit copied withSHA PASS. Report:outputs/sharpa_early_warning_dynamic_threshold/20261006_224500/threshold_audit.md.

Loss audit 2026-10-07T13:36:10.930305+08:00: project ProcVLM venv CPU numpy/matplotlib; histories50refit150fold plotted, no training; {"refit": {"count": 50, "minimum_at_last": 49, "tail_descending": 49}, "folds": {"count": 150, "minimum_at_last": 145, "tail_descending": 146}}. Per-update train loss only, no saved per-epoch held-out loss. Weekly copySHA PASS.

Training schedule audit 2026-10-07T13:42:33.066706+08:00: project ProcVLM venv stdlib,20 output snapshots summarized;30/patience8 mostly valBA, learned thresholds20fixedupdates. No training. Report WeeklySummary/10.5/10.5training_schedule_audit.md.
