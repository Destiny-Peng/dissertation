# Localization: separate failure and success Run sources

- Verified at: 2026-10-02T22:55:56.249583+08:00
- HEAD before uncommitted edits: `0d00867e9d7b5a64b00f55fd4ad2fe2a91365f46`
- Backend environment: `conda_envs/LF3R-robo-dopamine/bin/python`; CPU only.
- API environment: system `python3`; HTTP test uses a fake trainer that writes fixture files.
- Frontend: Node v24.19.0. No real inference/training, production jobs, or WebUI restart.

## Behavior

Analysis Localization Lab now offers **Failure / shared Run** and **Success Run** inputs, with completed Robo-Dopamine Runs suggested from the existing baseline catalog. Paths can also be pasted.

Set population to `failure_success` and select separate Runs to mix their cohorts using `success_ratio`. Each rollout keeps its own Run as the source of all fused, perspective and latent features. Saved annotations define terminal failure versus clean success; recovered success is excluded from the separate success cohort. Existing selected-mode coverage and native frame/latent token checks still apply.

An empty Success Run preserves the original shared source / historical pool behavior. Old saved presets remain valid. A positive success ratio with no usable annotated clean-success records raises an error before training, preventing experiments with empty success pools. The existing ratio sampler still caps the number selected by availability; actual selections are recorded in `training_records.json`.

`data_preflight.json` records both source paths, cohort sizes, exclusions and alignment. Output metadata records the separate success source and selection mode. Sources remain fixed across configurations to keep comparisons on the same pool.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_localization_sources tools.tests.test_robo_localization_specs tools.tests.test_robo_latent.LatentTests.test_aligned_training_records_load_same_run_latents tools.tests.test_robo_latent.LatentTests.test_missing_annotations_are_reported_before_training
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_localization_lab_preset_and_spec_job
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/localization-lab.js
/home/linxia/.nvm/versions/node/v24.19.0/bin/node /tmp/lf3r_localization_source_ui_check.js
python3 -m py_compile tools/robo_localization_head/specs.py tools/robo_localization_head/spec_runner.py tools/lf3r_annotator/analysis_localization_challenge.py
git diff --check
```

Results: 19 CPU unit checks and 1 HTTP integration check passed. Frontend VM checks passed for separate source round-trip, legacy defaults and catalog filtering. Python/JavaScript syntax and whitespace checks passed. No visual browser or actual model training verification was performed.
