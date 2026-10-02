# Unified latent input comparison preset

Verified: 2026-10-02 21:05 Asia/Singapore. Base commit: 825ff53a03fe1b124f2ec860f9f4d1d63adf7b1e, main. Python: system python3. JavaScript: existing Node v24.19.0. Existing unrelated repair and Run-filter changes were preserved. No training, inference, GPU allocation, or server restart was performed.

The seven standalone built-ins are replaced by one `latent_input_default` preset (label: Latent input comparison, PCA-64 default). It uses the existing coupled variants format. Base input is latent with PCA-64; variants list PCA-64 first, then PCA-32 and PCA-128, each with latent and latent+fused, followed by one fused-progress baseline. There are seven distinct configurations, five repeats by default (35 training runs). Each input variant can be edited or removed in the UI. Fused progress does not use PCA.

The UI previously only read/wrote target variants. It now also renders signal-mode/PCA input variants and preserves them through preset loading, editing and spec serialization. Existing target variants retain their previous behavior. Other built-in presets remain unchanged; saved experiment specs can still use the same signal modes and PCA dimensions.

Commands from PROJECT_ROOT:

```bash
python3 -m unittest tools.tests.test_robo_localization_specs tools.lf3r_annotator.tests.test_server.ServerTest.test_robo_latent_presets_options_and_status -v
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/localization-lab.js
git diff --check
```

All 13 tests passed in 0.531 seconds. Log: logs/latent_preset_merge_tests_20261002.log. Validation confirms normalized specs expand to seven unique configs, fused appears once, the first config uses 64D, and the API lists one unified preset without the old standalone names. An inline Node VM harness exercised the real input/target variant editor functions: all seven variants round-tripped exactly, PCA edits propagated, invalid PCA dimensions were rejected, and an existing hard-target variant round-tripped unchanged.

Refresh the browser to load the updated preset/editor. To run only the default 64D latent model, retain the `pca64_latent` variant and remove the others; to compare 64D latent, latent+fused and fused, retain those three variants. Changes to the base PCA field do not override explicit dimensions in variant rows; edit the rows for comparison settings.
