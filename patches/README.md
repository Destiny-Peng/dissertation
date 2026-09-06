# LF3R upstream migration patches

This directory stores small, intentional changes made in an upstream clone.
The upstream repository itself remains ignored by the LF3R root repository.

## ProcVLM

- Base commit: `377523a31f05bab9c0db5ac8b9edfa7b7f03968a`
- Patch: `ProcVLM-gpu-memory-utilization.patch`

After cloning ProcVLM on another workstation, check out the recorded base
commit and apply the patch from the LF3R root:

```bash
git -C repos/ProcVLM checkout 377523a31f05bab9c0db5ac8b9edfa7b7f03968a
git -C repos/ProcVLM apply ../../patches/ProcVLM-gpu-memory-utilization.patch
git -C repos/ProcVLM diff --check
```

The patch adds the optional `--gpu_memory_utilization` argument to ProcVLM's
inference entry point and forwards it to the vLLM engine. It does not include
model checkpoints, generated outputs, datasets, or Python environments.

## safe-openvla

- Base commit: `300dce26d44f407c725695d16cd445755c92cbd1`
- Patch: `safe-openvla-resolution.patch`

The LF3R natural-rollout wrapper passes `--render_resolution` and
`--record_resolution` to the official evaluator. Apply the safe-openvla patch
after cloning the recorded base commit:

```bash
git -C repos/safe-openvla checkout 300dce26d44f407c725695d16cd445755c92cbd1
git -C repos/safe-openvla apply ../../patches/safe-openvla-resolution.patch
git -C repos/safe-openvla diff --check
```

This patch adds those two evaluator fields, uses `render_resolution` for the
simulator camera, and stores `record_resolution` frames in replay videos while
leaving the 224x224 policy preprocessing unchanged.
