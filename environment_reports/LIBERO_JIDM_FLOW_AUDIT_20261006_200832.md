# LIBERO J-IDM offline flow audit

```json
{
  "checked_at": "2026-10-06T20:08:32.983249+08:00",
  "completed_rollouts": 82,
  "total_rollouts": 500,
  "total_adjacent_pairs": 137590,
  "offline_flow_construction": true,
  "per_pair_process_spawn": false,
  "window_frames": 4,
  "window_stride": 3,
  "pairs_per_full_window": 3,
  "cuda": true,
  "bf16_autocast": true,
  "gpu_workers": [
    0,
    1,
    2
  ],
  "complete_rollout_skip": true,
  "cache": "RGB + flow + low_dim trajectory in official-compatible per-rollout packed NPZ",
  "flow_codec": "qint8_zstd_npz",
  "training_loader": "vera/datasets/core/view_loader.py load_flow -> decode_packed_flow_frame",
  "example": "datasets/libero_jidm_training/20261006_123234/episodes/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo__demo_0.npz"
}
```

Evidence: tools/vera_libero/pack_training.py (GPU/bfloat16,4-frame windows,3-frame stride,window batch,OS locks and completed-pack skip); official reference repos/VERA/scripts/data/pack_pusht.py; training decode repos/VERA/vera/datasets/core/view_loader.py and packed.py. Each worker loads MegaFlow once; each epoch reads cached flow and derives original normalized action from cached low-dimensional states. Interruption recomputes only an unfinished rollout, not completed rollouts. The earlier13885 pairs refer to the zero-shot50-demo evaluation; full construction covers137590 pairs/500demos/two cameras.
