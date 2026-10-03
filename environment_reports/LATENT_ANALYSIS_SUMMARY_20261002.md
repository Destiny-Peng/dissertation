# Latest latent analysis results — 2026-10-02

Six completed experiments, submitted 21:54–22:23 Asia/Singapore. All use Robo-Dopamine source run `robo_dopamine-batch-7d3f81b2ba4b/robo_dopamine_20261002_185915_550373`, hard targets, BCE, five model-seed repeats and fixed split seed 42. The source has 135 usable rollouts, zero clean successes; 131 eligible failures are split 91/19/21. Rates below are means across five repeats; ± values are population standard deviations across model seeds, not confidence intervals. MAE is distance to the nearest failure interval in sampled sequence indices; each sample step corresponds to 8 native frames. Main hit rate counts any failure interval; first-event hit rate is reported separately.

| Job/time | Hidden/layers | Best latent configuration | Hit ± SD | First-event hit | MAE samples | Matched fused hit / MAE |
|---|---|---|---:|---:|---:|---|
| f47e55989e83 / 21:54 | 16/1 | PCA-32 latent + fused | 66.67% ± 6.02% | 64.76% | 2.619 | 74.29% / 0.800 |
| 6667c0188fed / 21:56 | 32/1 | PCA-32 latent | 71.43% ± 5.22% | 71.43% | 1.790 | 69.52% / 1.076 |
| 3b0e97d54c3c / 21:59 | 64/2 | PCA-64 latent + fused | 67.62% ± 4.67% | 60.00% | 1.990 | 66.67% / 1.181 |
| ad181b2ec06d / 22:03 | 128/2 | PCA-32 latent + fused | 70.48% ± 3.56% | 70.48% | 1.429 | 52.38% / 1.581 |
| fdd218e988bc / 22:10 | 256/4 | PCA-256 latent + fused | 55.24% ± 3.81% | 50.48% | 3.000 | not included |
| 19ecd5860157 / 22:23 | 64/1 | PCA-64 latent + fused | 68.57% ± 3.81% | 63.81% | 2.838 | not included |

The strongest tested fused baseline is hidden=16/layers=1: 74.29% hit, 67.62% first-event hit, MAE=0.800. The strongest latent hit is PCA-32 latent, hidden=32/layers=1: 71.43%, first-event=71.43%, MAE=1.790. Thus no latent configuration tested exceeds the highest fused hit rate, although first-event behavior and matched-architecture comparisons differ.

PCA-32/64 perform better than the tested PCA-128/256 configurations. For hidden=128/layers=2, PCA-32 latent+fused reaches 70.48% hit and first-event hit with MAE=1.429; PCA-64 latent+fused ties hit rate but has MAE=1.800 and first-event hit=66.67%. For the PCA-256 sweep, hit rates span 44.76–55.24%, despite hidden sizes 128/256 and 2/4 layers. Increasing model size/depth or PCA dimension has not produced a consistent gain.

Latest job 19ecd5860157 sweeps success ratios 0.1/0.2/0.5. All 60 training records contain empty success_train_ids. Metrics for each latent configuration are identical across ratios. This is not evidence that success ratio has no effect; the sweep added zero success examples.

The manually stopped job 56632bce731b and earlier failed jobs are excluded. Test population is only 21 rollouts; one rollout is 4.76 percentage points per repeat. Repeats vary model seeds on the same split, so they do not measure uncertainty across dataset partitions. Differences of 1–2 points should not be treated as established gains.

Detailed configuration results and source files:

## analysis-localization-f47e55989e83

Source summary: `outputs/robo_localization/20261002_135433_latent_input_default_55989e83/summary.csv`
Spec sweep: `[]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 64 | 16/1 | 0 | 58.10% ± 1.90% | 53.33% | 3.029 |
| s01_c002 | robodopamine_latent_plus_fused | 64 | 16/1 | 0 | 57.14% ± 3.01% | 49.52% | 4.438 |
| s01_c003 | robodopamine_latent | 32 | 16/1 | 0 | 60.00% ± 6.46% | 59.05% | 2.229 |
| s01_c004 | robodopamine_latent_plus_fused | 32 | 16/1 | 0 | 66.67% ± 6.02% | 64.76% | 2.619 |
| s01_c005 | robodopamine_latent | 128 | 16/1 | 0 | 56.19% ± 8.73% | 55.24% | 3.714 |
| s01_c006 | robodopamine_latent_plus_fused | 128 | 16/1 | 0 | 55.24% ± 7.74% | 53.33% | 3.895 |
| s01_c007 | fused | 64 | 16/1 | 0 | 74.29% ± 4.86% | 67.62% | 0.800 |

## analysis-localization-6667c0188fed

Source summary: `outputs/robo_localization/20261002_135631_latent_input_default_c0188fed/summary.csv`
Spec sweep: `[]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 64 | 32/1 | 0 | 53.33% ± 9.23% | 46.67% | 3.276 |
| s01_c002 | robodopamine_latent_plus_fused | 64 | 32/1 | 0 | 68.57% ± 4.86% | 58.10% | 2.114 |
| s01_c003 | robodopamine_latent | 32 | 32/1 | 0 | 71.43% ± 5.22% | 71.43% | 1.790 |
| s01_c004 | robodopamine_latent_plus_fused | 32 | 32/1 | 0 | 65.71% ± 7.00% | 62.86% | 1.924 |
| s01_c005 | robodopamine_latent | 128 | 32/1 | 0 | 52.38% ± 7.97% | 51.43% | 3.657 |
| s01_c006 | robodopamine_latent_plus_fused | 128 | 32/1 | 0 | 56.19% ± 5.55% | 53.33% | 3.200 |
| s01_c007 | fused | 64 | 32/1 | 0 | 69.52% ± 8.30% | 61.90% | 1.076 |

## analysis-localization-3b0e97d54c3c

Source summary: `outputs/robo_localization/20261002_135928_latent_input_default_97d54c3c/summary.csv`
Spec sweep: `[]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 64 | 64/2 | 0 | 61.90% ± 5.22% | 55.24% | 2.410 |
| s01_c002 | robodopamine_latent_plus_fused | 64 | 64/2 | 0 | 67.62% ± 4.67% | 60.00% | 1.990 |
| s01_c003 | robodopamine_latent | 32 | 64/2 | 0 | 58.10% ± 3.56% | 58.10% | 1.924 |
| s01_c004 | robodopamine_latent_plus_fused | 32 | 64/2 | 0 | 60.00% ± 5.71% | 59.05% | 1.790 |
| s01_c005 | robodopamine_latent | 128 | 64/2 | 0 | 50.48% ± 6.46% | 45.71% | 6.257 |
| s01_c006 | robodopamine_latent_plus_fused | 128 | 64/2 | 0 | 58.10% ± 4.67% | 51.43% | 4.524 |
| s01_c007 | fused | 64 | 64/2 | 0 | 66.67% ± 6.02% | 60.00% | 1.181 |

## analysis-localization-ad181b2ec06d

Source summary: `outputs/robo_localization/20261002_140312_latent_input_default_1b2ec06d/summary.csv`
Spec sweep: `[]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 64 | 128/2 | 0 | 66.67% ± 3.01% | 60.95% | 1.629 |
| s01_c002 | robodopamine_latent_plus_fused | 64 | 128/2 | 0 | 70.48% ± 5.55% | 66.67% | 1.800 |
| s01_c003 | robodopamine_latent | 32 | 128/2 | 0 | 64.76% ± 8.30% | 64.76% | 1.676 |
| s01_c004 | robodopamine_latent_plus_fused | 32 | 128/2 | 0 | 70.48% ± 3.56% | 70.48% | 1.429 |
| s01_c005 | robodopamine_latent | 128 | 128/2 | 0 | 56.19% ± 8.19% | 54.29% | 3.352 |
| s01_c006 | robodopamine_latent_plus_fused | 128 | 128/2 | 0 | 68.57% ± 4.86% | 64.76% | 2.686 |
| s01_c007 | fused | 64 | 128/2 | 0 | 52.38% ± 7.97% | 51.43% | 1.581 |

## analysis-localization-fdd218e988bc

Source summary: `outputs/robo_localization/20261002_141005_latent_input_default_18e988bc/summary.csv`
Spec sweep: `[{"path": "model.hidden", "values": [128, 256]}, {"path": "model.num_layers", "values": [2, 4]}]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 256 | 128/2 | 0 | 49.52% ± 7.74% | 43.81% | 6.562 |
| s01_c002 | robodopamine_latent_plus_fused | 256 | 128/2 | 0 | 44.76% ± 6.46% | 37.14% | 6.095 |
| s01_c003 | robodopamine_latent | 256 | 128/4 | 0 | 54.29% ± 3.81% | 53.33% | 2.800 |
| s01_c004 | robodopamine_latent_plus_fused | 256 | 128/4 | 0 | 53.33% ± 6.32% | 52.38% | 3.371 |
| s01_c005 | robodopamine_latent | 256 | 256/2 | 0 | 44.76% ± 9.33% | 40.95% | 5.952 |
| s01_c006 | robodopamine_latent_plus_fused | 256 | 256/2 | 0 | 52.38% ± 3.01% | 47.62% | 5.305 |
| s01_c007 | robodopamine_latent | 256 | 256/4 | 0 | 54.29% ± 6.46% | 52.38% | 3.210 |
| s01_c008 | robodopamine_latent_plus_fused | 256 | 256/4 | 0 | 55.24% ± 3.81% | 50.48% | 3.000 |

## analysis-localization-19ecd5860157

Source summary: `outputs/robo_localization/20261002_142314_latent_input_default_d5860157/summary.csv`
Spec sweep: `[{"path": "data.success_ratio", "values": [0.1, 0.2, 0.5]}]`

| Config | Signal | PCA | Hidden/layers | Success ratio | Hit ± SD | First-event hit | MAE |
|---|---|---:|---|---:|---:|---:|---:|
| s01_c001 | robodopamine_latent | 64 | 64/1 | 0.1 | 66.67% ± 5.22% | 60.00% | 2.571 |
| s01_c002 | robodopamine_latent_plus_fused | 64 | 64/1 | 0.1 | 68.57% ± 3.81% | 63.81% | 2.838 |
| s01_c003 | robodopamine_latent | 128 | 64/1 | 0.1 | 59.05% ± 3.81% | 54.29% | 3.657 |
| s01_c004 | robodopamine_latent_plus_fused | 128 | 64/1 | 0.1 | 59.05% ± 4.86% | 53.33% | 3.381 |
| s01_c005 | robodopamine_latent | 64 | 64/1 | 0.2 | 66.67% ± 5.22% | 60.00% | 2.571 |
| s01_c006 | robodopamine_latent_plus_fused | 64 | 64/1 | 0.2 | 68.57% ± 3.81% | 63.81% | 2.838 |
| s01_c007 | robodopamine_latent | 128 | 64/1 | 0.2 | 59.05% ± 3.81% | 54.29% | 3.657 |
| s01_c008 | robodopamine_latent_plus_fused | 128 | 64/1 | 0.2 | 59.05% ± 4.86% | 53.33% | 3.381 |
| s01_c009 | robodopamine_latent | 64 | 64/1 | 0.5 | 66.67% ± 5.22% | 60.00% | 2.571 |
| s01_c010 | robodopamine_latent_plus_fused | 64 | 64/1 | 0.5 | 68.57% ± 3.81% | 63.81% | 2.838 |
| s01_c011 | robodopamine_latent | 128 | 64/1 | 0.5 | 59.05% ± 3.81% | 54.29% | 3.657 |
| s01_c012 | robodopamine_latent_plus_fused | 128 | 64/1 | 0.5 | 59.05% ± 4.86% | 53.33% | 3.381 |
