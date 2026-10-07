# LIBERO IDM training failure

2026-10-07T21:12:53.128933+08:00

200 demos /58594pairs verified,157train/19validation/24test. Third training attempt, microbatch2 and accumulation8, completed30optimizer updates then a DataLoader worker aborted. Log reports CUDA initialization error in CUDA tensor destruction; fork-related inherited CUDA state is a hypothesis, not proven. No checkpoint exists (configured first save at200steps), so the30updates cannot be resumed.

Prepared fix: configurable multiprocessing_context=spawn, replace lambda worker initializer with picklable functools.partial; preview config outputs/vera-libero-training/subset200_batch2_20261007_200758/next_libero_idm_spawn.yaml. Source syntax checked; no GPU retry or runtime validation performed. Input/output/scales/architecture unchanged.

BLOCKED after3 training failures per AGENTS.md. Log: outputs/vera-libero-training/subset200_batch2_20261007_200758/training.log.
