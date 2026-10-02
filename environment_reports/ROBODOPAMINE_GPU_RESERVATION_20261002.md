# Robo-Dopamine GPU reservation verification

Date: 2026-10-02, Asia/Singapore. Branch: main. Base commit: 1f276bf1dabc5c9397d5f80b3a65371e87cd8642. Verification was performed on the working-tree changes before commit. Existing conflict resolutions and unrelated repair changes were preserved.

Environment: system python3 for CPU unit tests and WebUI; PROJECT_ROOT/conda_envs/LF3R-robo-dopamine/bin/python for CUDA and installed vLLM. No model inference or training was started.

Verification commands (from PROJECT_ROOT):

```bash
python3 -m unittest discover -s tools/baselines -p test_gpu_memory_reservation.py -v
python3 tools/baselines/test_robo_dopamine_persistent.py
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_robo_memory_reservation_options_reach_runner -v
/home/linxia/.nvm/versions/node/v24.19.0/bin/node --check tools/lf3r_annotator/static/results/run-config.js
git diff --check
python3 tools/baselines/run_lf3r_baseline.py --baseline robo_dopamine --manifest cache/lf3r_annotator/manifests/catalog-54b6ceb36ffbf512.jsonl --data-root . --output-dir outputs/baselines/reservation_dry_test --partition all --dataset-role libero_10 --limit 1 --gpu 2 --robo-reserve-gpu-memory --robo-reserve-mib 12288 --dry-run
```

Results: 9 reservation unit tests passed, existing persistent-worker tests passed, Run option validation/command construction test passed, JavaScript syntax and whitespace checks passed. Dry run saved both reservation options in run.json without starting a GPU worker. CPU tests cover TP budgets, fixed size and OOM cleanup, cancellation, authenticated handoff, actual runner budget propagation and failure finalization; these use mocks rather than a loaded vLLM model.

One small sequential CUDA allocation smoke was completed on GPU 2. The test set reserve_mib=64 and handoff_buffer_mib=0 solely to avoid reserving GiB during verification. It confirmed approximately 63.9184 MiB tensor allocation, keeper PID appearing in nvidia-smi (614 MiB including CUDA context), socket handoff, exit code 0 and removal of the keeper GPU process. The successful warm-cache attempt took approximately 1.254 seconds. The first two attempts exposed buffered-stdin daemon finalization and slow PyTorch interpreter finalization; fixes use unbuffered os.read for owner EOF and direct process exit after cleanup. The third attempt passed. Log: logs/gpu_reservation_smoke.log.

Full model loading/inference was not run. The handoff releases reserved tensors immediately before vLLM GPU initialization; it cannot provide exclusive GPU ownership or atomic cross-process memory transfer. PyTorch import/CUDA setup may still be slow on cold shared storage before reservation becomes ready. See tools/baselines/GPU_MEMORY_RESERVATION.md for usage and sizing.

Installed-vLLM compatibility check passed on 2026-10-02 at approximately 16:43: importing `gpu_reservation_worker.ReservedGPUWorker` with the Robo-Dopamine environment and verifying it is a subclass of `vllm.v1.worker.gpu_worker.Worker`. No CUDA tensors or model were initialized by this import check. Shared-storage imports and interpreter shutdown were slow; the same process was allowed to finish rather than restarted.

WebUI restarted in tmux session `lf3r-annotator`; server PID 2734881, port 8765. Startup log reported application initialization in 3.102 seconds and total startup 3.104 seconds.

Final live checks: GET / and GET /api/manifests both returned HTTP 200. Served HTML includes both reservation inputs. Nine CPU tests passed again after updating budget-history persistence and explicit keeper-failure propagation. Node was absent from the default shell PATH, so the existing Node v24.19.0 binary was used directly for the final syntax check.
