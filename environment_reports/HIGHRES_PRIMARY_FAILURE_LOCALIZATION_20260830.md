# LF3R High-Resolution Primary Failure Localization Report - 2026-08-30

## Result

A new temporal-analysis snapshot was generated from existing baseline raw outputs. No SAFE, ProcVLM, RynnValue, or Robo-Dopamine inference was started, and no raw baseline file was copied, rewritten, or used as a fallback.

Snapshot:

outputs/baseline_signal_analysis/highres_primary_20260830/

The snapshot contains 125 primary_natural labeled rollouts, 58 annotated events with an observable_onset_frame, 464 temporal event-signal rows, 4,472 clean-success pseudo-event signal rows, and 1,392 localization event rows across Q90/Q95/Q99.

## Inputs

| Method | Source | Selected coverage | Native sampling |
| --- | --- | ---: | --- |
| SAFE | outputs/baselines/full_136/safe_20260826_212113_540452 | 125/125 | all SAFE CSV rows |
| ProcVLM | outputs/baselines/full_136_01/procvlm_20260827_190222_580731 | 125/125 | dense raw frame records, max sampled frames 1024 |
| RynnValue | two web aggregate runs under outputs/baselines/web_runs | 124/125 | merged by rollout ID, no old-run fallback |
| Robo-Dopamine | outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444 | 125/125 | official frame interval 2 output |

The one missing RynnValue rollout is libero_10-task02-ep005-natural-e8fc18cf25. Its annotation is uncertain and has no failure event, so it contributes no event metric and remains an explicit unavailable rollout in method coverage. The 11 reference_natural rollouts are excluded from this primary comparison.

## Analysis environment and command

The analyzer ran with:

- interpreter: conda_envs/LF3R-ananlyse/bin/python;
- pinned environment: numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9;
- backend: MPLBACKEND=Agg;
- windows: pre 60 frames, post 60 frames, clean-background stride 30 frames.

All methods retain native sampling. No interpolation or cross-method resampling was performed. Native sample density differs materially: SAFE 164-520 samples per rollout, ProcVLM 164-520, RynnValue 42-131, and Robo-Dopamine 82-260.

## Q95 descriptive localization view

| Method / signal | Recall | Clean pseudo-event false alarm | Precision | F1 | AUROC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAFE / max_token_prob | 0.03 | 0.05 | 0.02 | 0.04 | 0.59 | 0.13 |
| SAFE / avg_token_prob | 0.03 | 0.05 | 0.02 | 0.04 | 0.56 | 0.10 |
| SAFE / max_token_entropy | 0.00 | 0.05 | n/a | n/a | 0.57 | 0.10 |
| SAFE / avg_token_entropy | 0.00 | 0.05 | n/a | n/a | 0.56 | 0.10 |
| ProcVLM / progress | 0.09 | 0.06 | 0.06 | 0.10 | 0.73 | 0.16 |
| RynnValue / value | 0.00 | 0.05 | n/a | n/a | 0.65 | 0.12 |
| Robo-Dopamine / progress | 0.24 | 0.05 | 0.12 | 0.28 | 0.88 | 0.34 |
| Robo-Dopamine / hop | 0.00 | 0.05 | n/a | n/a | 0.46 | 0.08 |

These values are event-level descriptive diagnostics. Thresholds are calibrated from clean-success pseudo-events in the same annotated collection; they are not independently tested detector operating points, causal claims, or evidence that one method is universally better. The UI and CSV files retain Q90 and Q99 to show the threshold trade-off.

## Generated artifacts

- metadata.json: selection, all source runs, source parameters, source manifest hashes, coverage, native alignment, and metric counts;
- method_coverage.csv: explicit missing-output coverage;
- event_metrics.jsonl and clean_background_metrics.jsonl: existing temporal metrics;
- localization_event_metrics.jsonl;
- localization_summary.csv;
- localization_by_failure_type.csv;
- localization_thresholds.csv;
- comparison_with_full_136_20260827.csv;
- REPORT.md and 504 per-rollout/grouped PNG plots.

The web GET /api/analysis endpoint now selects this snapshot as the latest complete result and returns compact localization tables/events without frame arrays. Older snapshots without localization files remain readable and report localization as unavailable.
