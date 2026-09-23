# Real-robot PKL conversion

`convert_realrobot_pkl.py` converts each transition PKL into one MP4 per camera view and baseline-ready JSONL manifests. By default, it extracts both `side_policy_256` and `wrist_1`. The source PKLs are never modified.

Each observation stores a two-frame temporal stack for each camera. The converter takes the latest history frame from every transition and appends the last transition's latest `next_observations` frame by default. The history dimension is not another camera. `state` is proprioceptive data and is not written as a video.

Run from the project root:

```bash
source project_env.sh
"$LF3R_SAFE_PYTHON" tools/convert_realrobot_pkl.py \
  --input outputs/realrobot/demo_buffer \
  --output outputs/realrobot/baseline_rollouts \
  --task "<task instruction>"
```

The default outputs are:

```text
outputs/realrobot/baseline_rollouts/videos/side_policy_256/*.mp4
outputs/realrobot/baseline_rollouts/videos/wrist_1/*.mp4
outputs/realrobot/baseline_rollouts/manifest.jsonl
outputs/realrobot/baseline_rollouts/manifest_side_policy_256.jsonl
outputs/realrobot/baseline_rollouts/manifest_wrist_1.jsonl
```

Each view manifest has a normal `video_path` for that camera, so it can be passed directly to the baseline runner. The canonical `manifest.jsonl` uses `side_policy_256` by default. To use the wrist camera with the canonical manifest, pass `--primary-camera wrist_1`; alternatively, use `manifest_wrist_1.jsonl` directly.

Videos are encoded as H.264 with FFmpeg's `libx264` by default (`yuv420p`, CRF 18). FFmpeg with `libx264` must be available on `PATH`. To explicitly use the old OpenCV MPEG-4 Part 2 output, pass `--video-codec mp4v`.

`--observation-keys` changes the camera fields to extract and requires at least two keys. For example:

```bash
"$LF3R_SAFE_PYTHON" tools/convert_realrobot_pkl.py \
  --input outputs/realrobot/demo_buffer \
  --output outputs/realrobot/baseline_rollouts \
  --observation-keys side_policy_256 wrist_1 \
  --primary-camera wrist_1 \
  --task "<task instruction>"
```

Use a view-specific manifest with a baseline, for example:

```bash
"$LF3R_SAFE_PYTHON" tools/baselines/run_lf3r_baseline.py \
  --baseline procvlm \
  --manifest outputs/realrobot/baseline_rollouts/manifest_wrist_1.jsonl \
  --data-root . \
  --partition natural_observation \
  ...
```

`--history-index` selects the frame within each camera's temporal stack and defaults to `-1` (latest). `--no-append-final-next` omits the final next-observation frame. `--fps` defaults to 30. Existing converted files are protected. Use `--resume` to keep files that already exist and rebuild the manifests; use `--overwrite` to regenerate videos and replace manifests.

This output is directly suitable for video baselines that consume `video_path` (ProcVLM, RynnValue, Robo-Dopamine, DenseReward). SAFE's trained detector consumes OpenVLA hidden-state pickles/CSV features, so this image-only conversion does not create SAFE latent inputs.
