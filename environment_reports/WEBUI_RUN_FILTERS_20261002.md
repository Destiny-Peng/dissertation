# WebUI Run rollout-filter verification

Timestamp: 2026-10-02 17:17–17:20 Asia/Singapore. Base commit: 825ff53a03fe1b124f2ec860f9f4d1d63adf7b1e (main). Environment: system Python 3.12 and existing Node v24.19.0. No inference or GPU allocation was started for verification. Unrelated repair changes were preserved.

From PROJECT_ROOT:

```bash
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_batch_rollout_filters_are_applied_before_worker_ranges tools.lf3r_annotator.tests.test_server.ServerTest.test_batch_rollout_filters_combine_with_missing_valid_coverage tools.lf3r_annotator.tests.test_server.ServerTest.test_variant_rollout_filters_use_source_annotation_and_outcome tools.lf3r_annotator.tests.test_server.ServerTest.test_batch_missing_valid_result_filter_skips_existing_parseable_outputs -v
python3 -m py_compile tools/lf3r_annotator/baseline_catalog.py tools/lf3r_annotator/baseline_jobs.py tools/lf3r_annotator/http_handler.py
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/runs/baseline-selection.js
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/runs/baseline.js
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/annotate/events.js
git diff --check
```

All four HTTP/backend tests passed in 4.684 seconds. Tests use temporary fake runners, covering annotated success overriding manifest failure, Complete+Success/Fail, All+Success, unreviewed manifest fallback, uncertain exclusion, explicit rollout IDs even for scope=all, worker ranges after filtering, invalid filters, missing-valid coverage, and source annotations/outcomes for instruction variants. Existing missing-valid behavior passed unchanged.

An inline Node VM smoke loaded the actual annotate/catalog.js and runs/baseline-selection.js. It confirmed selection for Complete+Success and Complete+Fail, filtered worker slicing, exclusion of another dataset_role, and coverage cache invalidation when filters change. Syntax, compilation and whitespace checks passed.

The live WebUI was not restarted, as explicitly requested by the user. The current baseline job remained running in its independent tmux session. New backend fields will become available after the user restarts WebUI; no change is applied to already running jobs.
