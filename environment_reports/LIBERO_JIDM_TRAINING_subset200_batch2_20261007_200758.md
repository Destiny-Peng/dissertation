# LIBERO J-IDM training

```json
{
  "status": "TRAINING_LAUNCHED",
  "updated_at": "2026-10-07T20:08:02.403196+08:00",
  "queue_pid": 658735,
  "training_started": true,
  "training_pid": 659066,
  "gpu": 0,
  "free_gpu_mib": 31911,
  "command": [
    "/mnt/hdd/qiuxia/pyr/LF3R/tools/run_libero_jidm_train.sh",
    "/mnt/hdd/qiuxia/pyr/LF3R/datasets/libero_jidm_training/20261006_123234_subset200",
    "experiment.training.max_steps=10000",
    "experiment.training.batch_size=2",
    "experiment.training.optim.accumulate_grad_batches=8",
    "experiment.training.enable_progress_bar=false",
    "experiment.validation.val_every_n_step=1600",
    "experiment.training.checkpointing.every_n_train_steps=200",
    "hydra.run.dir=/mnt/hdd/qiuxia/pyr/LF3R/outputs/vera-libero-training/subset200_batch2_20261007_200758",
    "+lf3r_observer._target_=vera_libero.training_observer.TrainingObserver",
    "+lf3r_observer.run_dir=/mnt/hdd/qiuxia/pyr/LF3R/outputs/vera-libero-training/subset200_batch2_20261007_200758",
    "+lf3r_observer.observe_steps=200"
  ],
  "max_optimizer_steps": 10000
}
```

Run: outputs/vera-libero-training/subset200_batch2_20261007_200758. Initial training health and ETA will be written to training_health.json and TRAINING_STATUS.md in the run directory.
