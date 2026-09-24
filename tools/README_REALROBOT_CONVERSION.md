# Real-robot PKL conversion

`convert_realrobot_pkl.py` converts each transition PKL into one MP4 per physical camera view and one LF3R manifest row per rollout. The source PKLs are never modified.

Each observation stores a two-frame temporal stack for each camera. The converter takes the latest history frame from every transition and appends the last transition's latest `next_observations` frame by default. The history dimension is not another camera. `state` is proprioceptive data and is not written as a video.

The current PKL format stores a binary `rewards` value on each transition. Any reward of `1` labels the rollout `success`; if all available rewards are `0`, it is labeled `failure`. If `rewards` is missing or contains non-binary values, the outcome is `unknown`.

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
```

Manifest schema v2 contains no top-level `video_path`. Each rollout has one explicit camera mapping, for example:

```json
{
  "schema_version": 2,
  "id": "realrobot-...",
  "camera_video_paths": {
    "cam_high": "outputs/realrobot/baseline_rollouts/videos/side_policy_256/realrobot-....mp4",
    "cam_wrist": "outputs/realrobot/baseline_rollouts/videos/wrist_1/realrobot-....mp4"
  }
}
```

Camera semantics come from the keys, not the MP4 filenames. By default `side_policy_256` is mapped to `cam_high` and `wrist_1` to `cam_wrist`. `--primary-camera` chooses which source observation is mapped to `cam_high`; it does not create another manifest or duplicate a video path.

Videos are encoded as H.264 with FFmpeg's `libx264` by default (`yuv420p`, CRF 18, `veryfast` preset). Frames are streamed directly to FFmpeg instead of building a second full-video buffer. To trade smaller files for slower encoding, pass `--video-preset medium`; `--video-preset ultrafast` favors speed at the cost of larger files. FFmpeg with `libx264` must be available on `PATH`. To explicitly use OpenCV MPEG-4 Part 2 output, pass `--video-codec mp4v`.

`--observation-keys` changes the physical camera fields to extract and requires at least two keys. For example:

```bash
"$LF3R_SAFE_PYTHON" tools/convert_realrobot_pkl.py \
  --input outputs/realrobot/demo_buffer \
  --output outputs/realrobot/baseline_rollouts \
  --observation-keys side_policy_256 wrist_1 \
  --primary-camera side_policy_256 \
  --task "<task instruction>"
```

Use the single generated manifest with any baseline:

```bash
"$LF3R_SAFE_PYTHON" tools/baselines/run_lf3r_baseline.py \
  --baseline procvlm \
  --manifest outputs/realrobot/baseline_rollouts/manifest.jsonl \
  --data-root . \
  --partition natural_observation \
  ...
```

Single-video baselines select the manifest's primary camera, preferring `cam_high`. Robo-Dopamine can consume the full camera mapping: a shared `cam_wrist` is adapted to both wrist input slots, while distinct `cam_left_wrist` / `cam_right_wrist` paths are used directly when present.

`--history-index` selects the frame within each camera's temporal stack and defaults to `-1` (latest). `--no-append-final-next` omits the final next-observation frame. `--fps` defaults to 30. Existing converted files are protected. Use `--resume` to keep videos that already exist and rebuild the manifest; reused files are marked in `camera_metadata`. Use `--overwrite` to regenerate videos.

SAFE's trained detector consumes OpenVLA hidden-state pickles/CSV features, so this image-only conversion does not create SAFE latent inputs.
