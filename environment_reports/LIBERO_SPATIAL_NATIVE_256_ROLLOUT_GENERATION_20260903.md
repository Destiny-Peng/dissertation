# LIBERO-Spatial Native 256x256 Rollout Pipeline

## Status

The native-resolution generation path is ready. No new rollout generation, GPU inference, or large download was started for this change.

## Existing resources reused

The following project-local resources were already present and were reused:

- LIBERO checkout: repos/LIBERO
- LIBERO-Spatial BDDL files: repos/LIBERO/libero/libero/bddl_files/libero_spatial
- LIBERO-Spatial initialization files: repos/LIBERO/libero/libero/init_files/libero_spatial
- Ten-task suite registration: repos/LIBERO/libero/libero/benchmark/libero_suite_task_map.py
- OpenVLA checkpoint: checkpoints/openvla-7b-finetuned-libero-spatial
- Existing OpenVLA environment: conda_envs/LF3R-openvla

The existing outputs/openvla_libero/lf3r-feasibility-natural/libero_spatial/ sample remains unchanged. Its inspected video is 224x224 and is treated as historical reference data, not as output from the new native path.

## Resolution contract

The official evaluator now accepts:

- render_resolution: simulator camera width and height;
- record_resolution: replay-video width and height.

The Spatial wrapper sets both to 256. The evaluator still computes the OpenVLA policy input with the official 224x224 get_image_resize_size path. It stores a separate native replay image from the 256x256 agent-view observation, so replay resolution does not change the model input.

LIBERO-10 keeps the existing 224x224 replay behavior for backward compatibility.

## New command-line path

The new wrapper is:

    bash tools/lf3r_annotator/generate_libero_spatial_native.sh 0 0 3 1 7 lf3r-data-natural-libero-spatial-256-manual

It reuses run_openvla_libero10_natural.py with --task-suite libero_spatial, the existing Spatial checkpoint, the official evaluator, and the existing memory-only GPU gate. GPU utilization is recorded but is not a blocking condition.

New output is isolated under:

    outputs/openvla_libero_spatial_native/<run-note>/libero_spatial/

The required run-note prefix is lf3r-data-natural-libero-spatial-256-. Existing output directories are never overwritten. On successful completion, the wrapper invokes build_manifest.py.

build_manifest.py now scans both OpenVLA output roots by default. Spatial natural records are classified as reference_natural; existing LIBERO-10 natural records remain primary_natural.

## Web path

The Review rollout-generation panel now has a Task suite / output selector:

- libero_10: existing LIBERO-10 output root and behavior;
- libero_spatial: native 256x256 simulator/replay output with 224x224 policy input.

POST /api/rollouts/generate accepts task_suite and returns suite, output-root, generator-script, and render/policy/record resolution metadata. The existing job status and log endpoints remain unchanged:

- GET /api/rollout-jobs/<job-id>
- GET /api/rollout-jobs/<job-id>/log

Generation remains a manifest/media writer and is mutually exclusive with baseline and temporal-analysis jobs. No automatic generation is performed by the web update.

## Verification

Completed without inference:

- inspected existing Spatial assets and checkpoint configuration;
- compiled the modified evaluator, suite-aware runner, manifest builder, server, and verifier;
- added fake-server coverage for Spatial suite selection, isolated output, job metadata, and manifest refresh;
- extended frontend contract coverage for the task-suite selector and parameter metadata.

The current manifest and old media were not rewritten during this implementation.
