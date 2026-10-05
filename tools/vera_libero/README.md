# Frozen VERA MimicGen J-IDM transfer evaluation

Run from the LF3R root with `source ./project_env.sh`. This pipeline evaluates a
frozen official MimicGen VGGT-Jacobian checkpoint on genuine adjacent images in
`datasets/libero_official/libero_10`, then executes the inferred actions in LIBERO.
It does not configure a video planner, generate future images, or train a model.

The experiment selects five equally spaced demo IDs per task (0,12,24,36,49),
covering 50 complete trajectories. `plan.json` registers every source hash,
trajectory, pair count, controller scale and fixed solver setting before inference.

```bash
VERA_LIBERO_GPU=1 ./tools/run_vera_libero.sh all
```

For an existing output directory, run or resume stages in order:

```bash
./tools/run_vera_libero.sh alignment "$EVAL_OUTPUT"
VERA_LIBERO_GPU=1 ./tools/run_vera_libero.sh predict "$EVAL_OUTPUT"
VERA_LIBERO_GPU=1 ./tools/run_vera_libero.sh playback "$EVAL_OUTPUT"
./tools/run_vera_libero.sh report "$EVAL_OUTPUT"
./tools/run_vera_libero.sh verify "$EVAL_OUTPUT"
```

For a new explicit directory, run `./tools/run_vera_libero.sh prepare "$EVAL_OUTPUT"`
first. Preparation also saves a diagnostic mapping GT achieved motion back to
OSC commands; that future-proprio diagnostic never enters predictions or playback.

`EVAL_OUTPUT` must be under LF3R. The wrapper derives all directories from
`project_env.sh`, uses separate J-IDM and existing OpenVLA/LIBERO Python environments,
and writes timestamped command logs. Stages execute sequentially on one GPU.
An interrupted prediction resumes at completed trajectories; alignment and playback
also preserve completed trajectories. The wrapper does not retry failed components.
Stop after three non-destructive failures as required by AGENTS.md.

Required assets:

- `checkpoints/vera-jidm/idm-mimicgen-285ouq1q/{model.ckpt,config.yaml,release_files.json}`
- `checkpoints/vera-jidm/cotracker3/scaled_offline.pth`
- official VERA, VGGT, CoTracker and LIBERO checkouts under `repos/`

The IDM is loaded strictly, verifies the published SHA256, and uses its bundled
VGGT weights without redownloading/overwriting the backbone. CoTracker3 offline
matches VERA's default serving tracker: two true frames, independent cameras,
15x15 query grid, official pretrained weights. Only current-frame images enter
the Jacobian network; CoTracker uses the current/next image pair.

Important contracts:

1. Processed LIBERO RGB[k] and proprio[k] are AFTER action[k]. Thus pair(k,k+1)
   corresponds to GT action[k+1] and restore state[k+1]. Every trajectory includes
   T-1 pairs; action[0] is excluded because its pre-action RGB is absent.
2. Camera order is agentview then wrist. Inputs are uint8/255 in the native
   OpenGL convention at 128x128. Recorded XML is restored with asset paths relocated
   only. Each demo validates sampled camera PSNR and pose/state alignment.
3. VERA learns achieved SE3 and finger position increments, not OSC commands.
   Use the checkpoint flow/action normstats and DatasetConfig scales (50/80).
   SE3 translation is `p1 - R_delta @ p0`, not `p1 - p0`; conversion back to OSC
   uses the current pose, configured 0.05m/0.5rad action ranges, and [-1,1] clipping.
4. Binary gripper commands are ambiguous from a stationary image pair. The fixed
   mapping uses the predicted finger sign, 0.18 normalized deadband and current
   finger position; no GT action or future proprio enters prediction.
5. Continuous execution starts both GT and IDM at the same recorded state and
   does not reset to future GT states. The separate one-step test restores the
   corresponding state before each predicted and GT action. Full simulator state,
   EEF motion error and final task success are saved. Video is sampled every five
   control steps for the first selected demo of each task.

This evaluates frozen J-IDM plus a fixed unit/interface mapping; it does not copy
the stack-specific VERA controller's zero roll/pitch, tuned gains, or online
adaptive controller. Model error versus achieved state deltas is saved separately
from error versus LIBERO GT commands. Continuous actions were inferred on recorded
images; this is not a closed-loop policy on deviating simulator images, and does
not establish reliability on WM-generated suffixes.

Outputs include `REPORT.md`, `summary.json`, `transfer.png`, `trajectories.csv`,
`model_load.json`, `provenance.json`, and per-demo `prediction.npz`, `actions.csv`,
`alignment.json`, playback arrays/metrics and sampled videos. The NPZ stores all
seven predicted/GT dimensions, signed/absolute/squared errors, original normalized
predictions and physical state deltas. Exact dataset/weight hashes and source
commits are recorded alongside results.

For LIBERO-specific training, preserve the original checkpoint's achieved-motion
target and normalization. Run `tools/run_libero_idm_target_conversion.sh` to export
all 500 LIBERO-10 success demonstrations into `datasets/libero_idm_targets/`.
It calls the upstream `SE3QuatDeltaAction` and `UnifiedDataset._derive_du` directly,
using observed EEF poses/finger positions, and retains associated OSC commands.
The export references unchanged RGB in the original HDF5 and copies the original
checkpoint config. It does not generate flow, select train/test splits or start
training; the original flow-supervision data still needs to be prepared.

`verify_results.py` checks every saved GT action against the original HDF5 at the
declared index, error arrays, full playback lengths and success flags, and decodes
the twenty comparison videos. Its findings are saved in `verification.json`.
