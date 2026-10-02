# Robo-Dopamine latent implementation verification

Date: 2026-10-01 (Asia/Singapore), final checks around 21:48 +08:00.
Base LF3R commit: `915286c`; changes remain in the working tree for review.

Implemented opt-in incremental hidden-state capture on the installed vLLM 0.11
V1 runner. The corrected selected position is the last token completing opening
`<score>`, whose hidden state predicts the next score token. No prompt or generation settings change.
Final decoder states and sample/before/after/token indices are saved in
`latent_features.npz`. Latent modes enforce the same fused-anchor run, rollout,
frame indices and incremental sample IDs.

`robodopamine_latent` and `robodopamine_latent_plus_fused` fit PCA inside each
repeat after splitting rollouts, using training data only. Default PCA output is
64D from 2560D, with 32/64/128D comparison presets. Checkpoints include fitted PCA and fit rollout IDs. Existing
checkpoint consumers reuse PCA. WebUI includes fused and 32/64/128D latent comparison presets,
manual extraction/re-inference controls and generated/missing status.

## Commands and results

Run from LF3R root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_latent tools.tests.test_robo_localization_specs -v
```

PASS: 19 tests. Generation is simulated; `core.train_bilstm` is mocked in the
training integration test. PCA fits and checkpoint serialization use only CPU
fixtures. No optimizer training or real model inference.

```bash
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_robo_latent_presets_options_and_status tools.lf3r_annotator.tests.test_server.ServerTest.test_robo_dopamine_web_command_defaults_to_fused -v
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_localization_lab_preset_and_spec_job -v
python3 tools/baselines/test_robo_dopamine_persistent.py
```

PASS: WebUI preset JSON, extraction command flag and missing/generated status;
existing fused CLI defaults; fake localization job lifecycle and artifact access;
persistent worker mock regressions (`ROBODOPAMINE_PERSISTENT_WORKER_TESTS_OK`).
Old fake-job fixtures needed raw string quoting and the artifacts now required
by the service. The worker fixture lacked `localization_checkpoint=None`.
Those fixtures were repaired before the checks passed.

```bash
python3 -m compileall -q repos/Robo-Dopamine/examples tools/robo_localization_head tools/lf3r_annotator tools/baselines/robo_dopamine_persistent_worker.py tools/baselines/robo_dopamine_runner.py tools/baselines/posthoc_robo_localization.py tools/baselines/run_lf3r_baseline.py
node --check tools/lf3r_annotator/static/localization-lab.js
node --check tools/lf3r_annotator/static/results/run-config.js
node --check tools/lf3r_annotator/static/results/view.js
python3 tools/baselines/run_lf3r_baseline.py --help
git diff --check
```

PASS: syntax, CLI exposes `--robo-extract-latent`, whitespace. Node used the
locally available v24.19.0 executable. Initial attempt in `LF3R-ananlyse` could
not import Torch, so CPU tests used the existing Robo-Dopamine Python instead.

The real GPU extraction path and real training remain for the user to run
manually. Instructions: `repos/Robo-Dopamine/LF3R_LATENT.md`.

## Token position correction

User clarified that the desired representation is the final token of the
opening `<score>` tag, before consuming the first score token. Defaults now
use `position=score_start`. Tests cover both a split opening tag and a single
tag token, decoder/logits alignment, and rejection of old `score_next` latents.
WebUI marks old-position artifacts as needing re-extraction.

Correction verified at 2026-10-01T21:59:35+08:00, current LF3R HEAD
`1f8a799` (working-tree changes): 8 CPU simulated latent tests and the
`test_robo_latent_presets_options_and_status` HTTP test passed using the same
commands above. Python compilation, `node --check` for results/view.js and
`git diff --check` also passed. No real inference or training was run.

## PCA preset expansion

At 2026-10-01T22:11:14+08:00 (HEAD `1f8a799`, working tree), added PCA-32 and
PCA-64 presets for both latent-only and latent+fused. Retained PCA-128 presets.
Schema, builder, PCA helper and training fallback now default to 64D. Updated
CPU tests exercise both feature modes at all three dimensions with training
mocked. The same 19 CPU tests and HTTP preset/status test passed, as did
JavaScript syntax and whitespace checks. No real inference or training.

## Failed run diagnosis and RPC fix

Log: `logs/baselines/web_runs/robo_dopamine_20261001_220819_084342.log`.
Its worker state records `fatal_engine_failure` during initialization, with
0 completed jobs and all 15 pending. Model loading, KV-cache allocation and
CUDA graph capture succeeded. The next call, `collective_rpc(install_capture)`,
failed because vLLM 0.11 safe Msgpack transport cannot serialize a Python
function. Previous simulated RPC tests did not cover this transport boundary.

Changed inference to pass `worker_extension_cls` as an importable class name
and invoke install/start/finish using named RPC methods. The extension executes
the capture hooks inside the worker. Returned vectors are converted to native
containers at the transport boundary, because untyped utility results cannot
restore ndarray type information with safe serialization. No insecure pickle
serialization environment override is required.

Verified 2026-10-01T22:24:07+08:00, HEAD `8ca036d`, working tree:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_latent -v
python3 -m py_compile repos/Robo-Dopamine/examples/inference.py repos/Robo-Dopamine/examples/latent.py
git diff --check
```

PASS: 9 tests, including original callable failure reproduction, named worker
extension resolution, real vLLM Msgpack request and typed UtilityOutput
round-trip with insecure serialization disabled, opt-in constructor setup,
simulated incremental generation, and mock-training PCA tests. No real model
was loaded and no inference, training or failed-run restart was performed.

## Actual latent analysis diagnosis and preflight

Verified 2026-10-01T23:01:36.104012+08:00, HEAD `db2a383` plus working tree changes.
User inference `robo_dopamine-batch-8774fe5a305d` completed 15 official
trajectories (`demo000`–`demo014`) with 547 aligned incremental score_start
latents, each 2560D. All 15 lack localization annotations.
Analysis `analysis-localization-87cdb3feba32` failed before training: historical
annotated fused anchors number 275, but their aligned latent count is zero.
The two data sets have no annotated latent overlap. Do not invent onset labels.

Actual CPU PCA validation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python tmp/check_robo_latent_analysis.py
```

PASS: 32/64/128D PCA fitted on 365 samples from 10 training rollouts only;
all 547 samples transformed with correct dimensions and finite values.
Held-out split: 2 validation and 3 test rollouts. This is PCA validation,
not a supervised localization training result. Details:
`tmp/robo_real_latent_pca_check.json`.

Added optional `base.data.source_run_root` and a corresponding builder field
for a single baseline run shared by all configurations. Empty retains the
historical pool. Single-run selection uses completed jobs in that run.
Training writes `data_preflight.json` before starting any model training.
Missing annotations and latent coverage are reported with corrective context.
The lower-level reader's opt-in `allow_empty` preserves existing callers'
error behavior while exposing exclusions to the preflight.
Actual 15-rollout preflight verified all exclusions as `annotation_missing`,
with audit at `tmp/robo_actual_run_preflight.json`.

Validation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_localization_specs tools.tests.test_robo_latent
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_robo_latent_presets_options_and_status tools.lf3r_annotator.tests.test_server.ServerTest.test_localization_lab_preset_and_spec_job
node --check tools/lf3r_annotator/static/localization-lab.js
git diff --check
```

PASS: 21 CPU tests and 2 HTTP tests (the HTTP job uses its existing fake
trainer). After the reader change, both preflight/alignment regression tests
were repeated. Actual localization training remains pending.

Prepared a cohort of 15 annotated terminal failures from libero_10 task00,
in `tmp/robo_latent_annotated_test_rollouts.json`. Inference command:

```bash
bash tmp/run_robo_latent_annotated_test.sh
```

Prepared `tmp/robo_latent_annotated_comparison_spec.json`: fused, PCA-64 latent,
PCA-64 latent+fused; five repeats; CPU, sequential configurations; user's
300 epochs/patience 50 and fixed rollout split preserved. Set source_run_root
to the resulting timestamped run directory before execution.
No new inference was launched: prior user preference required manual inference;
an explicit async authorization question is pending. No real localization
training has been reported as passing.

### Follow-up regression check (2026-10-01T23:04:18.258392+08:00)

Rechecked external state: exactly 15 latent artifacts remain, all from the
unannotated official batch; no annotated test inference directory exists.
Updated the existing alignment test's expected modes to include both new
latent modes, then ran:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_dopamine_localization_head -v
```

PASS: 13 existing regression tests (synthetic fixtures, not a real-data
localization training claim). Prepared the post-inference analysis command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python tmp/run_robo_latent_comparison.py <completed-run-path>
```

It validates the completed run and exact annotated cohort, writes a resolved
spec, runs the three configurations on CPU, and retains its log. Not executed
because the required annotated latent inference is still awaiting authorization.
