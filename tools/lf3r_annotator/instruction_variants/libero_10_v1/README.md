LIBERO-10 instruction-variant diagnostic manifest

This directory is a separate diagnostic dataset description generated from
datasets/lf3r_failure_rollouts/v1/manifest.jsonl. It does not replace or modify the source manifest,
videos, frame sidecars, annotations, or existing baseline outputs.

The source selection is task_suite=libero_10, dataset_role=primary_natural,
and analysis_partition=natural_observation. The source manifest hash is:

    e57cf4710ecb3c824dd93de6c5955c8cf1b242efc2d4606059acf7f7bc3e6980

The generated manifest contains one full_instruction row for every selected
rollout. Tasks 0-4 and 6-9 are compatible with two independent atomic goals,
so each of their rollouts also has subtask_a and subtask_b rows. Task 5 is
retained as full_instruction only: its official BDDL has one unique goal atom,
and splitting the surface wording into pick up versus place would not produce
two valid standalone LIBERO task instructions.

subtask_a and subtask_b are canonical labels based on source clause order or,
for task 8, official object identifier order (moka_pot_1 right then
moka_pot_2 left). They make no assumption about which subtask was executed
first in the original video.

Every row keeps the original video_path, task ID, episode ID, source rollout
ID, and original_full_instruction. The actual evaluator instruction is in
task_description and instruction. Counterfactual rows are explicitly marked
with instruction_type=counterfactual_single_subtask,
condition=subtask_a|subtask_b, and
analysis_partition=instruction_variant_diagnostic.

Future outputs must use the isolated namespaces:

    outputs/baselines/instruction_variants/libero_10/full_instruction/
    outputs/baselines/instruction_variants/libero_10/subtask_a/
    outputs/baselines/instruction_variants/libero_10/subtask_b/

The manifest is not consumed by the existing full-instruction analysis by
default. Validate the deterministic artifact with:

    python3 tools/prepare_libero10_instruction_variants.py --check-only

No baseline inference is run by this preparation step.
