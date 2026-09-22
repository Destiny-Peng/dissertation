# Localization Lab

Localization Lab is the unified failure-localization training and experiment-design
module for saved Robo-Dopamine fused progress/hop signals.

The current workflow is spec-driven. Experiments are no longer hard-coded as separate
Python runners.

## Core concepts

An experiment spec contains:

- a fixed base configuration;
- zero or more Cartesian sweep dimensions;
- a repeat count;
- optionally, sequential stages.

A stage runs its sweep, selects the best configuration, and passes that complete
configuration to the next stage. The default selector is highest in-interval rate,
then lowest MAE, then lowest MSE, but the primary metric and direction can be changed
from WebUI.

## WebUI

Open:

~~~text
Analysis -> Localization Lab
~~~

Localization Lab contains:

- **Experiment Builder**: fixed parameters, sweep dimensions, stages, run estimate,
  JSON preview, and launch;
- **Runs**: current job log and completed experiment outputs;
- **Presets**: built-in and project-local reusable experiment specs.

Project presets are stored under:

~~~text
config/robo_localization_presets/
~~~

Completed experiments are stored under:

~~~text
outputs/robo_localization/
~~~

## Built-in presets

- `bilstm_default`: one h16 / hard-label / BCE configuration.
- `label_loss_default`: staged label selection followed by loss selection.
- `success_ratio_default`: success-ratio x hidden-size Cartesian sweep.

## Experiment primitives

The implementation is intentionally split by research primitive:

- `core.py`: BiLSTM, rollout split, normalization, variable-length minibatching,
  packed sequence forward, optimization, early stopping, batched inference.
- `data.py`: failure-event and clean-success dataset construction.
- `targets.py`: hard and Gaussian multi-event targets.
- `losses.py`: BCE and temporal-softmax loss variants.
- `metrics.py`: interval localization metrics.
- `specs.py`: experiment schema, validation, sweep expansion, built-in presets.
- `spec_runner.py`: stage execution, repeated training, best-config inheritance,
  and artifact writing.

Robo-Dopamine inference is never rerun by Localization Lab.

## CLI

The WebUI writes an experiment spec and launches:

~~~bash
source ./project_env.sh

"$LF3R_ROBODOPAMINE_PYTHON" tools/train_robo_localization.py \
  --spec path/to/experiment_spec.json \
  --run-pool-root outputs/baselines \
  --output-dir outputs/robo_localization/manual_test
~~~

The runner selects the latest usable saved fused result independently for every rollout.

## Unified artifacts

Each completed experiment contains:

- `config.json`
- `metadata.json`
- `experiment_manifest.json`
- `training_records.json`
- `summary.csv`
- `per_rollout_predictions.csv`

`experiment_manifest.json` records the best configuration chosen by every stage.

## Supported sweep paths

The WebUI currently exposes convenient sweep controls for:

- training population / success ratio;
- complete target config, target kind, sigma pre/post, event-decay tau;
- BiLSTM hidden size;
- loss name and loss weights;
- batch size, learning rate, weight decay, and gradient clipping.

The JSON spec format itself is intentionally more general, so additional primitive
parameters can be exposed later without creating another experiment runner.
