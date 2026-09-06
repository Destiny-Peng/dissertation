# Annotator tmux persistent jobs — 2026-08-28

## Implemented behavior

- Baseline, temporal-analysis, and rollout-generation web commands are each
  launched in a project-owned tmux session named
  `lf3r-annotator-<job-id>`.
- Job metadata, wrapper, exit marker, and logs remain under the project:
  `logs/annotator_jobs/<job-id>/`.
- A new server process scans only recorded LF3R jobs, reattaches monitoring to
  active sessions, and marks a missing session as failed while preserving its
  log. It does not attach to unrelated tmux sessions.
- `/api/jobs` exposes the persistent list. The browser restores active jobs and
  keeps polling after a refresh.

## Concurrency

SAFE, ProcVLM, RynnValue, and Robo-Dopamine jobs are independently submitted
and may run concurrently. The GPU string is passed through exactly as supplied;
the service does not allocate devices or reject overlapping GPU IDs. Existing
free-memory checks remain in the runner, and GPU utilization is informational.

Rollout generation remains mutually exclusive with baseline and temporal
analysis because it writes media and rebuilds the manifest. Temporal analysis
may run beside baseline inference, provided its input run directories were
already complete and cover the selected rollout IDs.

## Verification

The fake-runner suite verified:

- persistent job JSON and tmux session metadata;
- stdout/stderr log capture and exit-code markers;
- successful completion and memory-blocked status mapping;
- missing-tmux failure without a normal background-process fallback;
- multiple baseline method submissions and manifest-writer conflicts;
- analysis interpreter readiness and `503` behavior when the dedicated
  environment is unavailable.

No real GPU inference was launched for this change.
