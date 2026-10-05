# Sharpa online one-sided Gaussian experiment

Completed 2026-10-05T18:11:56.412855+08:00; repository commit f46ce0111ed50472db2d4175786faf0ba8c66d22.

Environment: source project_env.sh, tools/run_trex.sh python using repos/ProcVLM/.venv/bin/python. Offline existing weights; GPU0 jobs sequential; frozen-feature probes only. No installs, original annotation edits, encoder finetuning, or changes to previous experiments. Disk free before work130GiB; GPU0 free25459MiB. Tag182000 is an output ID, not wall-clock start time.

Exact commands from repository root:

```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.gaussian_online prepare --source outputs/sharpa_merged_online_datasets/20261005_164000 --output outputs/sharpa_gaussian_online_data/20261005_182000 --device cuda:0 > logs/sharpa_gaussian_prepare_20261005_182000.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.gaussian_online train --source outputs/sharpa_gaussian_online_data/20261005_182000 --output outputs/sharpa_gaussian_online/20261005_182000 --device cuda:0 > logs/sharpa_gaussian_train_20261005_182000.log 2>&1
PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -m sharpa_tactile.report_gaussian_online --output outputs/sharpa_gaussian_online/20261005_182000 > logs/sharpa_gaussian_report_verified_20261005_182000.log 2>&1
```

Outcome COMPLETE. 114 mergedAlign and163 originalAlign interactions; no Insert targets; original80/17/17 rollout split. Natural distributions, no class augmentation. Unweighted soft-target CE; sigma in valid prediction ticks. 72 seed42 screen runs, selected exclusively by validation hard-GT BA;48 repeats yield5seeds per12selected configurations,120total. Training elapsed166.37seconds. All test hard GT identical across sigma, current frame position only, Key at Align end; annotated tails retained. Encoders frozen, continuous past16F6 once1280D, currentDeform2560D; GRU state retained within interaction. Source/checkpoint/cache hashes and split checked.

Verification PASS: all120 metrics recomputed, checkpoint and sigma selection checked, train-only normalization;12selected models CPU replay/future perturbation/prefix/streaming/reset checks. CPU/GPU maximum probability difference 0.00041711, atol5e-4/rtol1e-3; ALL predicted classes identical. Initial stricter CPU tolerance failed due numerical backend differences (logs/sharpa_gaussian_report_20261005_182000.log); diagnosed all12models, then verified with explicit bounds plus identical argmax; no model/results changed. Representative interval-prefix F6 encoding independently replayed on CPU (data/preflight.json).

Artifacts: [original training README](../outputs/sharpa_gaussian_online/20261005_182000/README.md), [data README](../outputs/sharpa_gaussian_online_data/20261005_182000/README.md), [weekly copy](../WeeklySummary/10.5/gaussian_online/README.md). Weekly copy SHA256 verified. All per-class metrics, count+normalized confusion, sigma sweep and deterministic probability curves retained. Failure recognition and seed variation remain substantial; no claim of reliable online outcome detection or fusion advantage.

Code hashes:
- tools/sharpa_tactile/gaussian_online.py: fde26024a2f2deb1636829629a9da553393e240a0b80633769d28b527a5001dc
- tools/sharpa_tactile/gaussian_targets.py: 58774978eefc2db0b3fab5987c4dd7163fac866f91b5dae62b2015e94b918a93
- tools/sharpa_tactile/report_gaussian_online.py: 150bf8a8f81b187dde345155fde6862287051c2a8fa517bc8c3676139c6d9b54
