# Robo-Dopamine BiLSTM success-negative ablation

This experiment uses only saved Robo-Dopamine fused progress/hop. It never reruns
Robo-Dopamine. The default evaluation source is the complete historical
Robo-Dopamine output pool: for each rollout, the newest usable fused result is
selected independently.

## Question

How does the amount of clean successful all-negative training data affect
failure localization, especially early false localization before the annotated
failure interval?

## Controlled comparison

The 0x failure-only baseline is always run. Non-zero success-to-failure ratios
are configurable with `--success-ratios` and default to `0.5,1,2`.

For example, with 25 failure training rollouts, the default target counts are
0, 12, 25, and 50 success rollouts. A custom run such as
`--success-ratios 0.25,0.5,1.5` runs 0x, 0.25x, 0.5x, and 1.5x.

Failure rollouts use the existing interval target:

- positive inside `[causal_onset, observable_onset]`
- negative everywhere else

Clean-success rollouts are negative at every timestep.

Validation and test contain failure rollouts only. Success trajectories are used
only as extra training negatives.

For a clean ablation, the following are computed from the failure-training
rollouts and then shared by both settings:

- fused progress/hop normalization
- positive-class weight

The BiLSTM seed, optimizer, learning rate, weight decay, epochs, patience,
gradient clipping, failure train/validation/test split, and evaluation protocol
are also identical between the two settings.

## Models

Only two models are trained:

- `tiny_bilstm_h16`: one bidirectional LSTM layer, hidden 16 per direction
- `tiny_bilstm_h32`: one bidirectional LSTM layer, hidden 32 per direction

Input is the complete native sequence `T x 2`:

- fused progress
- fused hop

The output is one score per native timestep and the predicted cut is the
rollout-global `argmax(score)`.

The implementation uses PyTorch. No linear probe, MLP, CNN, handcrafted
detector, extra loss, or new model family is trained in this experiment.

## Success rollout selection

For each repeat, the script creates one deterministic random ordering of
eligible clean successes. Same-task successes are shuffled first, followed by a
shuffled fallback pool from other allowed tasks. The 0.5x, 1x, and 2x settings
take nested prefixes from this same ordering, so increasing the ratio does not
replace previously selected success rollouts.

For task-held-out evaluation, successes from the held-out task are always
excluded. If fewer successes exist than requested, all available rollouts are
used and both requested and actual counts are reported.

There is no timestep-level split. Rollout IDs are never shared between failure
train/validation/test partitions.

## Metrics

The same interval-error definition is retained:

- prediction before causal onset: `prediction - causal`
- prediction inside the interval: `0`
- prediction after observable onset: `prediction - observable`

Reported metrics:

- in-interval rate
- within +/-1 / +/-3 / +/-5 samples
- before-interval rate
- after-interval rate
- median absolute interval error
- MAE
- MSE

Every repeat also reports the number of failure and clean-success training
rollouts.

## Run

Use a project-local environment that already has PyTorch. The Robo-Dopamine
environment is the intended default:

~~~bash
source ./project_env.sh

"$LF3R_ROBODOPAMINE_PYTHON" tools/train_robo_dopamine_localization_head.py \
  --run-pool-root outputs/baselines \
  --device auto
~~~

`--device auto` uses CUDA when available and otherwise falls back to CPU.

Defaults preserve the current BiLSTM training setup:

- hidden: 16 and 32
- epochs: 300
- patience: 35
- learning rate: 0.003
- weight decay: 1e-4
- gradient clipping: 5
- repeats: 5
- success ratios: 0.5, 1, 2 (configurable; 0x always included)

## Outputs

- `ablation_comparison.csv`: aggregate metrics for 0x / 0.5x / 1x / 2x
- `ablation_delta.csv`: each non-zero ratio compared directly against 0x
- `per_split_metrics.csv`: repeat-level metrics and failure/success train counts
- `per_rollout_predictions.csv`
- `task_held_out_metrics.csv`
- `task_held_out_summary.csv`
- `training_records.json`: exact rollout IDs used for each training run
- `split_manifest.json`
- `conclusion.md`
- `metadata.json`

The primary diagnostic is the shape of `before_interval_rate` across the four
ratios: whether a small amount of success helps and too much hurts, or whether
performance degrades immediately once success negatives are introduced.


## Result-pool selection

With `--run-pool-root`, the evaluator recursively discovers completed
full-instruction Robo-Dopamine runs. For each rollout ID it orders completed
candidates by that rollout's `worker_result.json` modification time and selects
the newest candidate with a usable fused signal. If the newest completed
candidate lacks a fused output, it falls back to the next newest usable fused
candidate for that rollout.

This means a newly rerun partial batch is automatically merged with older
results for rollouts that were not rerun. `--run-root` remains available only
for legacy single-directory evaluation.
