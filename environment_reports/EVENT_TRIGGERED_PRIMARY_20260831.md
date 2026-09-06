# LF3R Event-Triggered Primary Analysis - 2026-08-31

## Result

A complementary event-triggered analysis was generated from existing high-resolution baseline raw outputs and the current observable-onset annotations. No SAFE, ProcVLM, RynnValue, or Robo-Dopamine inference was started, and no raw baseline, rollout, media, or annotation file was modified.

- Snapshot: outputs/baseline_signal_analysis/event_triggered_primary_20260831/
- Selection: 125 primary_natural labeled rollouts
- Observable events: 58 total - 48 terminal failures, 9 recovered successes, 1 uncertain
- Clean-success control rollouts: 59
- SAFE: 125/125 signal rollouts
- ProcVLM: 125/125 signal rollouts
- Robo-Dopamine: 125/125 signal rollouts
- RynnValue: 124/125 signal rollouts
- RynnValue missing rollout: libero_10-task02-ep005-natural-e8fc18cf25

## Method

Every event is aligned at observable_onset_frame = 0. The event curves contain raw and per-instance onset-pre-window normalized median/IQR values at native sample offsets. The local change-score curves use failure-oriented native right-window-minus-left-window medians at 8, 16, 32, and 64 frame half-window scales, divided by the instance robust pre-onset scale.

Each event receives a clean-success pseudo-event anchor at the same normalized trajectory phase. Matching uses exact task_suite + task_id first, then same-suite and global fallback. Match level and control rollout are recorded in event_triggered_controls.csv.

No interpolation, resampling, or synthetic frame values are used. The median native interval is 1 frame for SAFE and ProcVLM, 2 frames for Robo-Dopamine, and 4 frames for RynnValue. This is a temporal sampling-density comparison, not evidence that all methods observe identical time grids.

## Interpretation

event_triggered_summary.csv reports the strongest aggregate case-versus-control separation for every method/signal/event group/representation/scale, its exact before/at/after-onset phase, and the median event-level peak lag. event_triggered_peak_events.csv retains rollout, task, failure type, peak offset, and matched control metadata. These are descriptive diagnostics on the annotated collection and should not be interpreted as independently validated detector performance or causal localization.

## Verification

The dedicated environment is conda_envs/LF3R-ananlyse/bin/python with MPLBACKEND=Agg. The generated snapshot contains the seven CSV tables, 16 PNG plots, metadata.json, and this analysis report. The Analysis API discovers it by its exact metadata analysis name and exposes compact tabular rows without frame arrays.
