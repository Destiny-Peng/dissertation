# Robo-Dopamine fused-hop failure analysis

This module performs CPU-only post-processing over already saved Robo-Dopamine
fused-progress/hop outputs and existing LF3R human annotations. It does not
launch Robo-Dopamine inference.

The existing detector sweep remains the source of detector configs. Detector
masks are evaluated on the unchanged native fused-hop grid; no interpolation,
new detector family, or new threshold search is introduced.

## Offline interval localization

Failure localization uses the annotated interval

`[causal_onset_frame, observable_onset_frame]`

as ground truth. For each failed rollout and existing detector config, the
original detector is scanned from rollout start to end and every positive
episode start is retained as a localization candidate. Windowed detector
timestamps therefore remain the original window-end timestamps.

For every candidate sample `s`:

`score(s) = median(hop[s-L:s]) - median(hop[s:s+R])`

with `L,R ∈ {2,3,5}`. Candidates without a complete left/right score window
remain visible in diagnostics but cannot be selected for that L/R pair. The
offline output is the maximum-score candidate; exact score ties choose the
earliest candidate.

The report compares this selector with:

- the original first detector trigger;
- the earliest native sample attaining the rollout-global maximum fused progress.

Signed interval error is measured in native samples: predictions before causal
onset are negative distance to causal onset, predictions inside the interval
have zero error, and predictions after observable onset are positive distance
from observable onset.

Reported localization metrics are in-interval rate, within ±1/±3/±5 samples,
before/after interval, median signed error, median absolute error, MAE, MSE, and
trigger coverage. They are reported separately for the first eligible event per
failed rollout (primary), all eligible failure events, and grasp failures. No
clean-FPR constraint is applied to this localization ranking, and ±10/±20 are
not localization metrics.

The main localization artifacts are:

- `interval_localization_ranking.csv`
- `offline_localization_diagnostics.csv`

The diagnostic table includes candidate count, scored-candidate count,
candidates before/inside/after the GT interval, first trigger, selected trigger,
selected score, and global fused-progress argmax for every event/config/L/R.

Run from the repository root:

```bash
source ./project_env.sh
python3 tools/analyze_robo_dopamine_incremental_hop.py \
  --run-root outputs/baselines/<completed-fused-robo-run>
```

Use `--selection` to restrict a superset run to requested rollout IDs and
`--no-plots` to skip plotting. Existing legacy sweep/recovery artifacts are
still written for compatibility, but the interval-localization view is based on
the offline post-processing described above.
