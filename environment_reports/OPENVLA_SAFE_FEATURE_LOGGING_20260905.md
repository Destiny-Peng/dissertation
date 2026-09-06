# SAFE-compatible OpenVLA feature logging

Date: 2026-09-05

## Result

The LF3R-owned rollout wrapper now has an opt-in --log-safe-features mode.
The ordinary path still forwards output_hidden_states=False. The opt-in path
forwards output_hidden_states=True to the existing official evaluator, then
postprocesses the evaluator artifacts. No SAFE feature extraction code was
added to the upstream evaluator.

The shell entry point also accepts --log-safe-features as its optional
seventh argument.

## Official representation

The local SAFE-compatible OpenVLA evaluator uses the official extraction:

    generated_outputs["hidden_states"][token][-1][0, -1, :]

For every policy call, the seven generated action-token hidden states are
taken from the last transformer layer and the final sequence position. The
official per-episode pickle therefore has shape:

    (T, 7, 4096)

The per-timestep raw feature contains 7 x 4096 values. SAFE's current loader
casts this tensor to float and applies token_idx_rel. Its default token_idx_rel
is 1.0, which selects the final generated action token and produces shape
(T, 4096).

The official evaluator's eposide_idx typo is accepted by the LF3R
postprocessor. The source .pkl is never rewritten.

## Output layout

For each completed episode under the evaluator's suite directory:

- task...succ*.pkl: official SAFE artifact, consumed directly by
  repos/SAFE/failure_prob/data/openvla.py.
- task...succ*.csv: official action/timestep log.
- task...succ*.mp4: replay video.
- task...succ*.safe_features.npz: compressed numeric sidecar containing
  hidden_states (T, 7, 4096), policy_step_index, environment_timestep,
  frame_index, actions (T, 7), task_id, episode_idx, and episode_success.
- task...succ*.safe_features.json: lightweight provenance and alignment
  metadata.

The sidecar has one row per policy decision. Its action row is copied from
the official CSV after normalization and gripper inversion, immediately before
env.step. Metadata records checkpoint, model family, sample count, center-crop
and image resolutions, selection rule, layer/position, source dtypes, numeric
dtype, shapes, git commit/dirty state, and count checks. Dense tensors are not
placed in JSON.

## Validation

Passed:

- SAFE extraction unit test: final layer/final sequence position and all seven
  token rows match the official rule.
- bfloat16 conversion test: official bfloat16 tensors convert to finite float32
  sidecar arrays without relying on a NumPy view of a Torch bfloat16 tensor.
- postprocessor fixture test: hidden-state count, action count, replay-frame
  count, shape, finite values, metadata, and unchanged source pickle.
- wrapper/shell syntax and contract tests.
- full annotator test suite after fixing a pre-existing tmux exit-marker race:
  43 tests passed.
- direct SAFE utility validation: loading the same official pkl and applying
  failure_prob.data.utils.process_tensor_idx_rel(..., 1.0) produced finite
  shape (80, 4096) training features.
- read-only validation of an existing clean official artifact:
  outputs/openvla_libero/lf3r-feasibility-natural/libero_spatial/
  task0--ep0--succ1.pkl has shape (80, 7, 4096), source dtype bfloat16,
  80 action rows, 80 replay frames, and finite values. This validates the
  official artifact and alignment without starting inference.

The requested fresh on/off rollout pair was not started. At the validation
check, GPU status was:

    GPU 0: 100% util, 69189 / 97887 MiB used, 28100 MiB free
    GPU 1: 100% util, 50004 / 97887 MiB used, 47285 MiB free
    GPU 2: 100% util, 94256 / 97887 MiB used, 3033 MiB free

The existing LF3R memory-only gate requires at least 30 GiB free and less than
50% total memory used. GPU 0 and GPU 2 fail both/most conditions; GPU 1 has
enough free memory but is just above the 50% used threshold. No GPU inference
was launched. Thus action-sequence equality and success equality between fresh
logging-off/logging-on runs remain runtime-pending, although the wrapper does
not modify sampling, preprocessing, action postprocessing, termination, or
success code paths.

## Compatibility and limitations

The official .pkl remains the direct SAFE training input, so no conversion is
needed for the existing SAFE loader. The .npz is the compact aligned export
for LF3R analysis and downstream tooling; it is not silently substituted for
the SAFE loader's .pkl convention.

The current upstream evaluator already exposes output_hidden_states; it does
not expose a save_safe_features configuration field. LF3R therefore performs
the sidecar step after evaluation. The only existing upstream diff remains the
pre-existing LF3R native replay-resolution change, not this feature logger.
