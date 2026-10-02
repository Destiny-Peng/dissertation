# Run rollout filters

The Run page has independent **Annotation status** and **Rollout outcome** selectors. Both default to All, preserving the previous selection behavior.

- **Complete only** requires a saved annotation with `review_status=complete`. It is independent of whether a baseline result already exists.
- **Success only** / **Fail only** use the saved annotation `outcome_label` when present, otherwise the source rollout's manifest `ground_truth_outcome`. Uncertain outcomes are excluded by either selector.
- Filters combine with the selected dataset_role, instruction condition, and existing-result filter. Worker ranges are relative to the filtered list; changing either selector resets the total range and rebalances workers.
- **No valid baseline run result** still requires complete annotations and skips parseable existing results. Coverage queries use the same outcome/status filters.

The backend repeats filtering before constructing the runner's explicit rollout IDs. Jobs save `review_status_filter`, `outcome_filter`, and `scope_rollouts_before_rollout_filters`. Instruction variants use the source rollout's annotation and outcome.

API batch fields: `review_status_filter` is `all` or `complete`; `outcome_filter` is `all`, `success`, or `failure`. These are batch selection fields, not runner options. Restart the WebUI backend and refresh the browser after updating the code. Already running jobs keep their original selection.
