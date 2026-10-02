# Localization analysis PCA optimization

Verified: 2026-10-02 21:52 Asia/Singapore. Worktree: main; HEAD at final verification: 5ca82cec8f8c7712696c2e5dadd7ac9e6114919d. Unrelated changes and concurrent commits were preserved.

Stopped only analysis-localization-56632bce731b after verifying PID 3985624 command line contained both train_robo_localization.py and the requested job ID. SIGTERM was sent; wrapper recorded exit code 143 and supervisor finished. Its tmux session and worker exited. Existing checkpoint/output files remain under outputs/robo_localization/.web_jobs/analysis-localization-56632bce731b/output. The existing analysis supervisor labels a manual SIGTERM as failed, rather than cancelled; this is the user-requested termination, not a new runtime error.

Changes:

- Exact NumPy SVD is retained. One stage-local PCA cache is shared by config workers. The largest requested latent PCA dimension is fitted once per identical training matrix; smaller dimensions use the prefix of that exact basis.
- Cache keys contain training rollout IDs, per-rollout matrix shapes and SHA-256 of the actual training-only latent feature values. Validation/test data and appended fused signals do not enter the fit or key. Different training cohorts or changed latent values require a fresh fit. Concurrent identical requests wait for one in-flight fit; failures unblock waiters and permit retry.
- The CLI sets job-local OPENBLAS/MKL/OMP/NUMEXPR thread limits before importing NumPy/PyTorch. Default: one thread; optional LF3R_ANALYSIS_CPU_THREADS overrides this with a positive integer. No system/global environment was changed.
- Logs now include analysis_import_start, data_load_start/done, config submission, per-repeat prepare_start/done, pca_fit_start/done, pca_cache_wait/hit and training_start, with relevant shapes and elapsed times.

Verification used PROJECT_ROOT/conda_envs/LF3R-robo-dopamine/bin/python. Commands from PROJECT_ROOT:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_pca_cache tools.tests.test_robo_latent -v
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' conda_envs/LF3R-robo-dopamine/bin/python -m unittest tools.tests.test_robo_latent.LatentTests.test_training_integration_reuses_pca_across_fixed_splits_and_configs -v
CUDA_VISIBLE_DEVICES='' conda_envs/LF3R-robo-dopamine/bin/python tools/train_robo_localization.py --help
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 conda_envs/LF3R-robo-dopamine/bin/python cache/tmp/benchmark_robo_pca_20261002.py
python3 -m py_compile tools/robo_localization_head/latent.py tools/robo_localization_head/spec_runner.py tools/train_robo_localization.py tools/tests/test_robo_pca_cache.py tools/tests/test_robo_latent.py
git diff --check
```

Four new PCA-cache tests and ten existing latent tests passed (14 total, 163.320 seconds including slow vLLM dependency imports). The additional fixed-split integration test passed in 0.139 seconds: four configs x two repeats called the mocked trainer eight times but exact PCA fitting once. It exercised the shared cache through the configuration wrapper and checkpoint persistence. Existing changing-split tests still refit independently. Numerical tests compare cached exact components against direct SVD, and confirm unchanged downstream fused dimensions/checkpoint consumers. CLI, compilation and whitespace checks passed. No real training or inference was started; GPU was disabled for the actual-data benchmark.

Actual-data CPU benchmark: used training IDs from the stopped job's first checkpoint, 91 training rollouts and 5874 samples. With one BLAS thread, initial exact SVD fit for PCA-128 took 5.8638 seconds; PCA-32/64/128 cache hits took 0.0302/0.0278/0.0280 seconds. This measures PCA preprocessing only, under the observed cache/storage conditions, and does not guarantee total job startup or training duration. Data loading, cold dependency imports and other contention remain separate costs.

Logs: logs/robo_pca_optimization_tests_20261002.log, logs/robo_pca_fixed_split_tests_20261002.log, logs/robo_pca_cli_help_20261002.log, logs/robo_pca_benchmark_20261002.log. Benchmark JSON: cache/tmp/robo_pca_benchmark_20261002.json. These transient artifacts are ignored. The analysis was not relaunched and WebUI was not restarted. New analysis jobs invoke the updated CLI directly.
