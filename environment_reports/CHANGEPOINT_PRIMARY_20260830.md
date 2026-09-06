# LF3R Local Change-Point Analysis Report - 2026-08-30

## Result

A new local change-point snapshot was generated from existing baseline raw outputs. No SAFE, ProcVLM, RynnValue, or Robo-Dopamine inference was started, and no raw baseline file was modified.

Snapshot: outputs/baseline_signal_analysis/changepoint_primary_20260830/

- selected rollouts: 125 primary_natural labeled rollouts;
- observable-onset events: 58;
- clean-success reference trajectories: 59;
- local half-window scales: 8, 16, 32, and 64 video frames;
- local features: level, variance, and slope;
- event metric rows: 16,704;
- summary rows: 1,152;
- failure-type summary rows: 1,728; threshold rows: 96;
- RynnValue coverage: 124/125, missing libero_10-task02-ep005-natural-e8fc18cf25;
- other method coverage: SAFE 125/125, ProcVLM 125/125, Robo-Dopamine 125/125.

The missing RynnValue rollout is uncertain and has no observable failure event. It remains unavailable and is not filled with the older low-sampling output. The 11 reference_natural rows are excluded.

## Inputs and environment

The four methods use the same high-resolution primary inputs as the current global analysis:

- SAFE: outputs/baselines/full_136/safe_20260826_212113_540452;
- ProcVLM: outputs/baselines/full_136_01/procvlm_20260827_190222_580731;
- RynnValue: the two aggregate web runs rynnvalue-batch-fe9e05c5ba54/...222438_749771 and rynnvalue-batch-cc8deb3f2e09/...221918_087826, merged by rollout ID;
- Robo-Dopamine: outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444.

The command used conda_envs/LF3R-ananlyse/bin/python with MPLBACKEND=Agg. The environment pins are numpy 2.2.6, pandas 2.3.3, and matplotlib 3.10.9. Analysis is CPU-only and does not trigger GPU inference.

## Method

For every native sample center and each scale, the analyzer compares native samples in [center-scale, center) with [center, center+scale):

- level: right-window median minus left-window median;
- variance: log ratio of right/left sample variance;
- slope: right-window least-squares slope minus left-window slope.

A robust median plus MAD/IQR scale is fitted to clean-success and same-rollout non-onset candidates. The unusual score is the absolute robust deviation; direction is deliberately not used to select the unusual change. Q90, Q95, and Q99 thresholds are calibrated from one trajectory-level maximum pseudo-event per clean-success rollout; same-rollout non-onset maxima remain a separate reference. Event windows are clipped by recovery, terminal failure, the next event, and trajectory bounds. Peak/first threshold crossing, signed lead/lag, absolute error, tolerance hits, misses, and post-identification direction are retained. Precision, F1, AUROC, and average precision use event peak scores as positives and clean-success pseudo-event peaks as negatives; these are descriptive ranking diagnostics. No interpolation or cross-method resampling is performed.

## Interpretation

At Q95, local unusual changes are generally sparse and often not close to the annotated observable onset. This supports using the snapshot to inspect signal-specific change-point behavior and sampling-scale sensitivity, but not to claim a validated detector. The report's full Q90/Q95/Q99 tables and failure-type breakdown should be read together with the trajectory-level clean-success and same-rollout non-onset false-alarm references.

## Comparison

comparison_with_full_136_20260827.csv contains:

- high-resolution global response, persistence, and recovery medians versus the legacy full_136_20260827 snapshot, with deltas;
- all Q95 local change-point recall, false-alarm, precision/F1, AUROC/AP, onset-near, peak-distance, and tolerance metrics;
- explicit blank legacy values for local metrics because the old snapshot has no equivalent local feature definition.

## Artifacts

- metadata.json
- method_coverage.csv
- changepoint_event_metrics.jsonl
- changepoint_summary.csv
- changepoint_reference_summary.csv
- changepoint_by_failure_type.csv
- changepoint_scales.csv
- localization_event_metrics.jsonl
- localization_summary.csv
- localization_by_failure_type.csv
- localization_thresholds.csv
- comparison_with_full_136_20260827.csv
- REPORT.md
- plots/changepoint_recall_q95_by_scale.png
- plots/changepoint_peak_distance_q95_by_scale.png
- plots/changepoint_onset_near_q95_by_scale.png

The old full_136_20260827 and highres_primary_20260830 snapshots remain unchanged.
