# WebUI lazy tactile index loading

Verified 2026-10-02T15:20:33.225132+08:00, HEAD `3a8e53c` plus working tree.

The tactile service now reads only failrecovery_manifest.jsonl at construction.
Frame/event JSONL files and camera/event indices load on the first request for
a rollout, then remain cached for frame, series, image and sprite endpoints.
Per-rollout locks prevent duplicate concurrent loads without blocking other
rollouts. Invalid episode data returns the existing not-found response without
breaking startup or caching a partial index; a later request can retry repaired data.

Validation:

```bash
python3 -m unittest tools.lf3r_annotator.tests.test_tactile_service -v
python3 -m py_compile tools/lf3r_annotator/tactile_service.py
git diff --check
```

PASS: 5 tests cover startup read scope, cache reuse across all four endpoints,
first access through each endpoint, concurrent same-episode and independent
episode loading, and corrupt-data recovery. No model inference or training.

Real-data initialization on this machine:

```json
{
  "tactile_init_seconds": 0.3296792069450021,
  "manifest_rollouts": 151,
  "jsonl_files_read": 1,
  "episode_indices_loaded": 0
}
```

The 151-rollout manifest took about 0.33 seconds to initialize, with exactly
one JSONL read and zero per-episode indices loaded. This measures tactile
initialization only and does not claim total WebUI startup is 0.33 seconds.

Updated WebUI launched with `bash tools/lf3r_annotator/run_server.sh 8765` in
tmux session `lf3r-annotator-lazy-tactile`. During startup, remaining disk waits
were observed in historical job-directory recovery and baseline_runs.sqlite3.
These are separate from tactile loading and were not changed by this patch.
