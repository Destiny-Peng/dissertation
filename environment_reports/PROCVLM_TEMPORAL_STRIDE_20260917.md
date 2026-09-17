# ProcVLM temporal-stride support

Date: 2026-09-17

## Scope

Added wrapper support for testing the temporal-scale mismatch between 30 FPS LIBERO videos and approximately 10 FPS training data. No GPU inference was run while implementing this change.

## Semantics

The existing target-frame selection and max-sampled-frames policy are unchanged. Frame stride controls only the spacing of source-video frames inside each target's temporal window.

With window size 8 and frame stride 1, the window is consecutive source frames ending at target t. With window size 8 and frame stride 3, the window is:

    [t-21, t-18, t-15, t-12, t-9, t-6, t-3, t]

At the beginning of a video, negative indices retain the existing clamp-to-frame-zero padding behavior. Each ProcVLM raw record now stores frame_stride and source-frame window_frame_indices.

The project runner and UI currently retain the established window-size default of 4 to avoid changing existing runs. For the requested temporal experiment, set window size to 8 explicitly and frame stride to 3. Frame stride 1 remains the default and preserves the normal consecutive-frame behavior for the dense short-video configuration.

## Interfaces

- Direct ProcVLM inference: --frame_stride N.
- Persistent ProcVLM worker: --frame-stride N.
- LF3R runner: --procvlm-frame-stride N.
- Annotator: Frame stride is available in both single-rollout and batch ProcVLM configuration.

## Validation

- Four pure sampling tests pass in repos/ProcVLM/tests/test_inference_sampling.py.
- Modified Python files compile with python3 -m py_compile.
- Existing parallel-runner tests pass.
- Focused server validation and command-forwarding test passes.
- Direct inference, persistent worker, and LF3R runner --help output expose the new option.

The full annotator discovery run still contains unrelated pre-existing frontend contract failures and one unavailable test module in the current checkout; these are documented separately from the new stride checks.
