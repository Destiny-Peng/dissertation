# LF3R Annotator instruction-condition view - 2026-09-08

## Result organization

The completed Robo-Dopamine aggregate run was read-only audited and explicitly classified as
the original full-instruction condition:

- run: outputs/baselines/web_runs/robo_dopamine-batch-326e663e25bc/robo_dopamine_20260908_105724_234057/
- status: complete
- selection: 166/166 rollouts
- completed/failed: 166/0
- source manifest: datasets/lf3r_failure_rollouts/v1/manifest.jsonl
- condition: full_instruction
- frame interval: 8
- evaluation: fused, with incremental/forward/backward component outputs
- raw output contents were not moved, copied, or modified

The run.json now records instruction_condition, instruction_variant, and the source manifest
explicitly. Existing older runs are still discovered by the server and classified from their
manifest path when possible.

## Review behavior

Review keeps one source rollout, video, annotation, and queue entry. The header selector
offers Full instruction, A, and B when a prepared diagnostic variant exists. A/B rows come
from:

    tools/lf3r_annotator/instruction_variants/libero_10_v1/manifest.jsonl

They reuse the source video only as a display reference. Their IDs are distinct
(source-id--subtask_a or source-id--subtask_b), and full-instruction raw directories are
never used for A/B. Until a separate baseline run is produced from a variant manifest, A/B
baseline cards show an explicit unavailable/condition-view-only state.

Tasks without two validated atomic goals expose only Full instruction. The A/B labels are
canonical goal labels, not assertions about temporal execution order in the observed video.

## Validation

- Variant artifact check: python3 tools/prepare_libero10_instruction_variants.py --check-only
- Server syntax: python3 -m py_compile tools/lf3r_annotator/server.py
- Focused baseline read test: passed
- Focused variant API test: passed
- Real project smoke: 166 rollouts loaded; primary task 0 exposes full/A/B; task 5 exposes full only; completed Robo-Dopamine run classified as full_instruction
- No GPU inference was started

The existing isolated health test can exceed its three-second HTTP test timeout while importing
the analysis environment dependencies; this is unrelated to the instruction-condition change.
