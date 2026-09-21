# Robo-Dopamine BiLSTM success-negative ablation

This experiment uses only saved Robo-Dopamine fused progress/hop. It never reruns
Robo-Dopamine.

## Question

Does adding clean successful trajectories as all-negative training data provide
useful hard negatives for failure localization, especially by reducing
predictions that occur before the annotated failure interval?

## Controlled comparison

Two training settings are run on exactly the same failure-rollout split:

1. `failure_only`
2. `failure_plus_success`

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

For each repeat, clean successes from tasks represented in the failure-training
split are preferred. If none exist, the script falls back to all eligible clean
successes.

For task-held-out evaluation, successes from the held-out task are always
excluded from training.

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

"$LF3R_ROBODOPAMINE_PYTHON"   tools/train_robo_dopamine_localization_head.py   --run-root outputs/baselines/<completed-fused-robo-run>   --device auto
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

## Outputs

- `ablation_comparison.csv`: aggregate failure-only vs failure+success metrics
- `ablation_delta.csv`: direct metric deltas for each hidden size
- `per_split_metrics.csv`: repeat-level metrics and failure/success train counts
- `per_rollout_predictions.csv`
- `task_held_out_metrics.csv`
- `task_held_out_summary.csv`
- `training_records.json`: exact rollout IDs used for each training run
- `split_manifest.json`
- `conclusion.md`
- `metadata.json`

The primary signal to inspect is whether `failure_plus_success` lowers
`before_interval_rate` without increasing late predictions or damaging
in-interval localization.
