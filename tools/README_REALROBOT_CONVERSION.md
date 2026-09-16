# Real-robot PKL conversion

`convert_realrobot_pkl.py` converts the NumPy transition files under `outputs/realrobot/demo_buffer/` into one MP4 per PKL plus a manifest JSONL. The manifest uses project-relative `video_path` entries and can be passed to the existing video baseline runner. Source PKLs are not modified.

The observed file layout is a list of transitions. Each transition contains `observations`, `next_observations`, `actions`, `rewards`, and status fields; `observations["side_policy_256"]` has shape `(2, H, W, 3)` and is a two-frame history stack. The converter takes the latest history frame from every transition, then appends the final transition's latest `next_observations` frame by default. This avoids writing overlapping history frames twice. It does not infer task semantics from robot state or actions.

Run from the project root:

```bash
source project_env.sh
python3 tools/convert_realrobot_pkl.py \
  --input outputs/realrobot/demo_buffer \
  --output outputs/realrobot/baseline_rollouts \
  --task "<task instruction>"
```

Output:

```text
outputs/realrobot/baseline_rollouts/manifest.jsonl
outputs/realrobot/baseline_rollouts/videos/*.mp4
```

Use the generated manifest with a video baseline, for example:

```bash
python3 tools/baselines/run_lf3r_baseline.py \
  --baseline procvlm \
  --manifest outputs/realrobot/baseline_rollouts/manifest.jsonl \
  --data-root . \
  --partition natural_observation \
  ...
```

The `--observation-key` option can select another camera key, such as `wrist_1`; `--history-index` selects a frame from the stack and defaults to `-1` (latest). `--no-append-final-next` omits the final next-observation frame. `--fps` defaults to 30. Existing converted files are protected. If a previous conversion stopped part way, use `--resume` to keep existing videos and rebuild the complete manifest; pass `--overwrite` explicitly to regenerate them.

This output is directly suitable for video baselines that consume `video_path` (ProcVLM, RynnValue, Robo-Dopamine, DenseReward). SAFE's trained detector consumes OpenVLA hidden-state pickles/CSV features, so this image-only conversion does not create SAFE latent inputs.
