# LIBERO-specific J-IDM data preparation

The original MimicGen model contract is retained: current RGB per camera at128x128,
7D normalized achieved SE3/finger motion, and dense flow supervision. No OSC command
regression head or affine calibration is introduced. The original checkpoint's
camera order, action/flow normalization and native wrist bottom80px mask are kept.

All500 LIBERO-10 success demos are split by complete trajectory:400 train,50 val,
50 test. Test IDs0,12,24,36,49 per task preserve the pre-registered zero-shot baseline;
five validation IDs per task are drawn seed42 from the remaining demos. This measures
new demonstrations of the same tasks, not unseen-task generalization. No normstats
are fitted on any LIBERO split.

Canonical packs live in episodes/. Each split has an index.json referencing its own
packs with project-local relative paths. Packed RGB uses lossless PNG (the official
PIL decoder accepts it), retaining source pixels exactly. Dense flow uses frozen
MegaFlow-flow weights, default952px internal feature width,4-frame windows with
one-frame overlap,8 refinement iterations and BF16 autocast. Resulting forward
flow is in128px image coordinates, quantized with the official qint8_zstd_npz codec.
The last RGB frame has no outgoing flow; official window sampling excludes it.
Motion-track supervision is not needed: original algorithm supervision=flow and
load_motion_tracks=false.

The upstream held-out dataset patch adds test_dataset routing and an opt-in finite
val/test length. It does not alter training sampling or the neural-network contract.
Patch: tools/patches/vera-jidm-heldout-datasets.patch.

From the LF3R root:

```bash
source ./project_env.sh
LIBERO_JIDM_GPU=1 tools/run_libero_jidm_prepare.sh worker
```

The current data pointer is cache/libero_jidm_training_current_run.txt. Packed files
are written atomically. Successful per-demo JSON records allow trajectory-level
resume. GPU stages and large downloads are sequential. Stop after three failures
per component, per AGENTS.md. The packer checks free disk before each trajectory.
The worker runs packing, verification and final reporting in sequence, resumes
completed packs, and records status/commands in worker_status.json and
worker_commands.json. Run only one worker. The first-demo smoke has already passed;
do not repeat it when resuming. Individual `pack` and `verify` stages are also
available for a stopped worker. At the measured shared-GPU rate, full default flow
generation takes approximately50 hours; the estimate varies with GPU load.

Verification checks disjoint splits, all packs and hashes, lossless RGB, all targets
against the prior official target export, flow finiteness/codec bounds and official
loader shapes. One frozen forward computes the official loss on a real8-frame,
2-camera batch; no backward, optimizer step or trainer.fit runs.

The generated config uses the released architecture and bundled backbone weights.
Native VGGT learned positional embeddings require518px model construction; runtime
images remain128px. Fine-tuning will start from official MimicGen weights with fresh
optimizer state, batch1/accumulate16, BF16, disabled W&B and project-local output.
Other training settings are inherited from the release; training duration and
validation protocol should be selected before a formal experiment.

The user explicitly authorized formal training on2026-10-06. The queued first
round runs10000 optimizer steps from the release weights, with a fresh optimizer
and accumulate16. Full-data verification must finish first; GPU preparation and
training remain sequential. It selects the GPU with the most available memory
(at least32GiB free) and waits for35GiB free disk to retain full checkpoints.
This authorization supersedes AGENTS.md's default prohibition on large training
for this run only.

The user subsequently authorized parallel GPU preprocessing on2026-10-06,
overriding the default sequential-GPU rule for preparation. Start/resume the
detached chain with `bash tools/run_libero_jidm_parallel.sh`. A double-forked,
session-independent supervisor is reparented to PID1; its PID/children/commands
are in pipeline.json under the authorized run. The previous tool-session
processes exited unexpectedly after16 completed demos, with stale status files;
their records are archived before replacement. Completed packs are reused.
The user further authorized larger preprocessing batches and GPU2. Current
defaults use GPUs0/1/2 with initial batch2/4/1 and unchanged952px/window4/8-iteration
MegaFlow settings. Workers share unfinished trajectories using nonblocking
OS file locks held through atomic packing and successful report writing, so
faster workers process more demos. Successful packs are skipped. They write
separate flow model records and disjoint trajectory files. Each worker stops
after at most three failures; retries halve its batch to a minimum of1.
These are initialization batches, not fixed limits. Each actual full4-frame
flow-window group records its allocated-memory peak above the idle model.
Workers account for globally free memory and reusable allocator cache, then
grow batch by at most2x within measured capacity, a15percent transient cushion
and2GiB headroom. Batch shrinks when capacity falls. The initial maximum is32;
retries halve this ceiling as well. Short final windows do not lower the full
window peak estimate. No extra benchmark frames are processed. Runtime caps
and headroom are controlled by batch_control.json without a restart. Actual
batch, peak/free memory and seconds/window are in batch_runtime.gpuN.json.
Larger batch does not guarantee better throughput on a GPU shared with other
jobs; throughput and memory are recorded rather than inferred from total VRAM.
Set LIBERO_JIDM_PREP_GPUS and LIBERO_JIDM_PREP_BATCHES together to override
the launch defaults. Verification starts after all GPU workers exit, followed by the
authorized training queue. Do not launch the sequential preparation worker
alongside this parallel chain. The earlier50-hour estimate is for one GPU;
parallel ETA must be updated from measured combined completed-pair throughput.

The active queue is recorded at cache/libero_jidm_training_authorized_run.txt.
Its queue_status.json reports the actual state; training has not started while
WAITING_FOR_VERIFIED_DATA. After launch, an opt-in official Lightning callback
observes200 optimizer steps, nonzero finite gradients, actual parameter updates,
one full50-demo held-out validation and checkpoint writing. Validation runs every
3200 microbatches (200 optimizer steps); checkpoints every200 optimizer steps.
It then writes training_health.json, TRAINING_STATUS.md and the environment report
with measured throughput/ETA and disables itself. It does not stop the trainer.
The queue only waits to reap the child after launch; it does not poll training.
Normal initial health is not a claim of action accuracy or simulator success.
Patch: tools/patches/vera-jidm-training-observer.patch. The main-entry changes
support a complete released YAML without Hydra experiment/dataset/algorithm groups;
the released SLURM flag is removed and local skip_download is enabled.

Only after preparation.json is READY and verification.json passes is this launch
entry available (it is NOT executed as part of preparation):

```bash
LIBERO_JIDM_GPU=1 tools/run_libero_jidm_train.sh
```

Achieved state motion is still different from the original OSC command. A separately
validated controller adapter remains necessary before predictions become reliable
LIBERO simulator actions. This preparation does not include WM or recovery trials.
