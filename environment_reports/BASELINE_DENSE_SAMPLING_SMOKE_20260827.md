# Dense-Sampling Baseline Smoke Test

日期：2026-08-27  
范围：2 条 primary-natural rollout；ProcVLM 与 Robo-Dopamine 各执行 2 条，GPU0 顺序执行。

## 配置

- ProcVLM：`--window_size 4`；未传 `--max_sampled_frames`，使用上游默认上限 512。
- Robo-Dopamine：`--frame-interval 4`。
- 两个 vLLM worker：目标为当前空闲显存的 `0.80`；runner 在启动前换算为 vLLM 的 total-memory fraction。
- GPU Util 仅记录，不作为 gate。

测试输入：

| Rollout | 视频源帧数 | 类型 |
| --- | ---: | --- |
| `libero_10-task00-ep000-natural-07acb763b6` | 414 | success |
| `libero_10-task00-ep001-natural-477e2f3af1` | 520 | failure |

## 结果

首次测试（14:49–14:51）误将 `0.80` 直接作为 vLLM 的 total-memory fraction，因此两种 baseline 均为 `0/2` 完成；失败发生在模型推理启动前的 vLLM 显存预检。该结果保留作为语义修正前的诊断记录：

| Baseline | Run metadata | 采样准备 | vLLM 结果 |
| --- | --- | --- | --- |
| ProcVLM | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/dense_sampling_smoke_20260827/procvlm_20260827_144945_475975/run.json) | 414→414；520→512 | 需要 76.01 GiB，free 70.38 GiB |
| Robo-Dopamine | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/dense_sampling_smoke_20260827/robo_dopamine_20260827_145107_704673/run.json) | worker 在 vLLM 初始化前失败 | 需要 76.01 GiB，free 70.26 GiB |

两次尝试都未生成模型 raw output；失败日志明确建议降低 memory utilization 或释放外部进程占用。监控期间 GPU0 used 约 24,668–25,223 MiB，free 约 72,066–72,621 MiB；vLLM 初始化时可用值进一步记录为 70.38/70.26 GiB。GPU0–2 均有外部 `sfe` Python/MPS 进程，未终止或修改。

## 可复现命令与后续

命令及每条 rollout 的返回码保存在各自的 `commands.jsonl` 与 `jobs.jsonl`；完整日志位于：

- `logs/baselines/dense_sampling_smoke_20260827/procvlm_20260827_144945_475975.log`
- `logs/baselines/dense_sampling_smoke_20260827/robo_dopamine_20260827_145107_704673.log`

上面的 76.01 GiB free 只对应首次将 `0.80` 错当作 vLLM total-memory fraction 的失败尝试，不是当前 runner 的显存要求。当前命令应使用 `--vllm-free-memory-fraction 0.80`；它会在每条 worker 启动前按实时 free/total 换算，修正后的复测见下节。


## 修正后复测

修正后，`0.80` 表示当前 free GPU memory 的比例。runner 在每条 worker 启动前读取 `nvidia-smi`，按所选 GPU 的最小 `free/total` 比例换算并记录实际传给 vLLM 的值。

| Baseline | Run metadata | 状态 | raw samples |
| --- | --- | --- | ---: |
| ProcVLM | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/free_memory_smoke_20260827/procvlm_20260827_160033_373567/run.json) | complete, 2/2 | 414、512 |
| Robo-Dopamine | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/free_memory_smoke_20260827/robo_dopamine_20260827_160349_222723/run.json) | complete, 2/2 | 104、130 |

两次运行记录的目标均为 `requested_free_fraction=0.80`。ProcVLM 实际传入 vLLM 的 total-memory fraction 为 `0.587142`；Robo-Dopamine 为 `0.587142` 和 `0.587616`。GPU0 监控观测到 used 最大 `82,091 MiB`、free 最小 `15,199 MiB`；GPU Util 仍只作观测，未作为 gate。修正后的 4 条 rollout 均生成了模型 raw output，满足本次 2 条 smoke test 的显存要求。

修正后的完整日志：

- [ProcVLM log](/mnt/hdd/qiuxia/pyr/LF3R/logs/baselines/free_memory_smoke_20260827/procvlm_20260827_160033_373567.log)
- [Robo-Dopamine log](/mnt/hdd/qiuxia/pyr/LF3R/logs/baselines/free_memory_smoke_20260827/robo_dopamine_20260827_160349_222723.log)
- GPU 监控：[ProcVLM CSV](/mnt/hdd/qiuxia/pyr/LF3R/logs/baselines/free_memory_smoke_20260827/procvlm_gpu.csv)、[Robo-Dopamine CSV](/mnt/hdd/qiuxia/pyr/LF3R/logs/baselines/free_memory_smoke_20260827/robo_dopamine_gpu.csv)
