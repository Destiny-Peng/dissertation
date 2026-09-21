# Lightweight Robo-Dopamine failure-localization head

This probe trains small temporal heads on **already saved** Robo-Dopamine fused
progress/hop. It never reruns Robo-Dopamine and does not modify the GRM.

## Target

The primary target is the first eligible event on each \`terminal_failure\`
rollout. Ground truth is the native-sample interval corresponding to

\`[causal_onset_frame, observable_onset_frame]\`.

Every native sample inside that interval is positive; all other samples in the
same failed rollout are negative. A trained head produces one score per native
sample and the final cut point is the rollout-global \`argmax(score)\`.

Using one primary interval per failed rollout keeps the training target
consistent with the one-cut-point output. Multi-event analysis remains a
separate diagnostic problem.

## Input

Each timestep receives a short symmetric offline context around fused progress
and fused hop. Default context radius is 3 native samples, so each example is a
\`2 x 7\` window. Edge samples use edge padding. Mean/std normalization is fit
from **training rollouts only**.

## Models

The default probe compares progressively:

1. \`linear_probe\`: weighted logistic regression;
2. \`tiny_mlp\`: one hidden ReLU layer with 16 units;
3. \`tiny_cnn\`: 8 learned 1D filters with kernel 3 and a position-aware readout.

All three are implemented with NumPy and run in the existing CPU Analysis
environment. No PyTorch/CUDA dependency is added.

A BiGRU is intentionally gated rather than added automatically. It is worth a
follow-up only if the CNN repeatedly improves over the linear/MLP probes,
indicating that learned temporal structure is helping.

## Splits

No timestep-level split is allowed.

For ordinary evaluation the script creates repeated rollout-level
train/validation/test splits, stratified as far as the small dataset permits by
task and failure type. Model fitting and feature normalization use train only;
early stopping and handcrafted-baseline selection use validation only; test is
untouched until final evaluation.

Learning-curve subsets are nested within each split and use 10, 20, 30 (when
available), and all train rollouts. Repeated rollout-level splits are aggregated
with mean and variance.

Task-held-out evaluation is also produced for tasks with at least two eligible
failed rollouts by default. The held-out task is used only as test data.

## Baselines

The learned heads are compared with:

- best existing handcrafted first-trigger rule;
- existing offline max-change-score changepoint baseline;
- earliest rollout-global fused-progress argmax.

Existing fused-hop analysis artifacts are reused when available. For the first
two baselines, the candidate rule/config is selected on validation rollouts and
then frozen for test. If compatible artifacts are missing, the script may
recompute the existing detector sweep from saved fused-hop signals only; this is
CPU post-processing and still does not rerun Robo-Dopamine.

## Metrics

All localization errors are in native samples relative to the GT interval:

- before causal onset: \`prediction - causal\`;
- inside interval: \`0\`;
- after observable onset: \`prediction - observable\`.

Reported metrics are in-interval rate, within +/-1 / +/-3 / +/-5 samples,
before/after interval, median absolute interval error, MAE, and MSE.

## Run

From the repository root:

~~~bash
source ./project_env.sh

conda_envs/LF3R-ananlyse/bin/python \
  tools/train_robo_dopamine_localization_head.py \
  --run-root outputs/baselines/<completed-fused-robo-run>
~~~

To bind comparison to a specific completed handcrafted/offline analysis, add:

~~~text
--baseline-analysis-dir outputs/robo_dopamine_incremental_hop/<run>
~~~

Default probe controls include \`--context-radius 3\`, \`--repeats 5\`,
\`--epochs 300\`, \`--patience 35\`, and \`--min-task-test-rollouts 2\`.

## Outputs

Each run writes:

- \`model_comparison.csv\`
- \`learning_curve.csv\`
- \`per_rollout_predictions.csv\`
- \`split_manifest.json\`
- \`per_split_metrics.csv\`
- \`task_held_out_metrics.csv\`
- \`task_held_out_summary.csv\`
- \`training_records.json\`
- \`conclusion.md\`
- \`metadata.json\`

\`conclusion.md\` intentionally answers only this probe question: whether the
lightweight temporal head beats the existing signal-level rules under strict
rollout/task splits, and whether the observed learning curve still looks
data-limited. Richer GRM hidden features are a next experiment, not part of this
one.
