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
git -C repos/ProcVLM apply patches/ProcVLM-gpu-memory-utilization.patch
git -C repos/ProcVLM diff --check
```

The patch adds the optional `--gpu_memory_utilization` argument to ProcVLM's
inference entry point and forwards it to the vLLM engine. It does not include
model checkpoints, generated outputs, datasets, or Python environments.
