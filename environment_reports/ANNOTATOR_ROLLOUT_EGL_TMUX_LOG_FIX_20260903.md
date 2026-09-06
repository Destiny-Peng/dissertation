# Annotator rollout EGL/tmux/log fix - 2026-09-03

## Diagnosis

The failed web generation job rollout-688ba5028732 was already launched by the
project TmuxJobSupervisor in session
lf3r-annotator-rollout-688ba5028732. Its persisted record and wrapper are under
logs/annotator_jobs/rollout-688ba5028732/. The failure occurred inside
mujoco.osmesa.GLContext with the PyOpenGL error
AttributeError: 'NoneType' object has no attribute 'glGetError'.

The problem was the rendering backend selected by both natural-rollout wrappers:
they exported MUJOCO_GL=osmesa and PYOPENGL_PLATFORM=osmesa.

## Changes

- generate_libero10_natural.sh and generate_libero_spatial_native.sh now use
  MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl.
- Web rollout generation explicitly passes the same EGL variables to
  TmuxJobSupervisor.submit(), so the generated project wrapper records the
  complete headless rendering environment.
- Rollout generation continues to use the same persistent tmux path as baseline
  jobs. It is monitored through /api/rollout-jobs/<job-id> and its persistent
  record includes the tmux session, wrapper, interpreter, log, and exit marker.
  No ordinary detached-process fallback is used.
- Persistent View log cards now preserve their open/closed state and fetched
  tail across task polling. Poll-driven card re-rendering no longer hides an
  expanded log; clicking the button still toggles it manually.

## Validation

- bash -n passed for both rollout wrappers.
- Python compilation passed for the annotator server, supervisor, pipeline verifier,
  temporal analyzer, and rollout runner.
- Fake tmux generation tests passed for LIBERO-10 and LIBERO-Spatial, including
  persistent job metadata, EGL wrapper exports, completion, manifest refresh, and
  log retrieval.
- Frontend contract tests passed for EGL markers and persistent log behavior.
- No real GPU inference, rollout generation, manifest rewrite, or user-output
  deletion was performed for this fix.
