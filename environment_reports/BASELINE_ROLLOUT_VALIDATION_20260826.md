# Representative Baseline Rollout Validation

日期：2026-08-26  
范围：8 条 representative natural-observation rollout；不执行全量 136 条评估。

## 结论

- selection：2 条 clean success、2 条 recovered success、4 条 failure。
- SAFE、ProcVLM、RynnValue、Robo-Dopamine 四个 baseline 均完成 8/8，失败 0。
- raw output、SAFE 特征公式、ProcVLM 进度解析、RynnValue value-head 捕获、Robo-Dopamine 官方 score 保真检查全部通过。
- GPU 采用 memory-only 门槛：不要求 Util 低于 100%，未终止或修改外部 MPS 工作负载。

## Representative selection

清单：[representative_rollouts_20260826.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/representative_rollouts_20260826.json)  
SHA-256：`c28f78a9bf7f6ee70e571ab0c00270a89490e644c0b1a4406ef27bb0fd6f81ea`

| 类别 | Rollout ID | 标注 failure type |
| --- | --- | --- |
| clean_success | `libero_spatial-task00-ep000-natural-be4a4090b6` | none_success |
| clean_success | `libero_10-task02-ep000-natural-d0c9300789` | none_success |
| recovered_success | `libero_10-task00-ep000-natural-5ef44c034f` | grasp_failure |
| recovered_success | `libero_10-task06-ep002-natural-888f529fd3` | collision |
| failure | `libero_10-task00-ep001-natural-477e2f3af1` | grasp_failure |
| failure | `libero_10-task03-ep007-natural-2dd9870cd4` | dropped_object |
| failure | `libero_10-task05-ep003-natural-fefb27649d` | placement_failure |
| failure | `libero_10-task08-ep002-natural-2bc533b685` | collision |

## Actual runs

| Baseline | Run metadata | 参数摘要 | 状态 |
| --- | --- | --- | --- |
| SAFE | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/representative_20260826/safe/safe_20260826_193421_272938/run.json) | official handcrafted features | complete, 8/8 |
| ProcVLM | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/representative_20260826/procvlm/procvlm_20260826_194717_717199/run.json) | window=4, sampled=16, max tokens=128, vLLM memory=0.55 | complete, 8/8 |
| RynnValue | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/representative_20260826/rynnvalue/rynnvalue_20260826_195300_710951/run.json) | frames=16, steps=16, batch=4, max tokens=128 | complete, 8/8 |
| Robo-Dopamine | [run.json](/mnt/hdd/qiuxia/pyr/LF3R/outputs/baselines/representative_20260826/robo_dopamine/robo_dopamine_20260826_203106_485078/run.json) | frame interval=15, batch=1, forward, vLLM memory=0.55 | complete, 8/8 |

四次运行均使用 GPU 0 顺序执行。最终只读 GPU 状态为：

```
GPU 0: 75368 MiB free / 97887 MiB total
GPU 1: 73294 MiB free / 97887 MiB total
GPU 2: 74335 MiB free / 97887 MiB total
```

Util 均为 100%，但没有作为 gate。已有外部 `sfe`/MPS 进程保持不变。

## Raw output 与 signal extraction

- SAFE：8 个 `safe_features.csv`，总计 3226 行。按源 CSV 重算 7-token probability/entropy 的 max、avg 及 expanding running mean，最大绝对误差 `7.772e-16`，属于 CSV 浮点序列化误差；shape、columns、timestep 和有限值检查通过。
- ProcVLM：8 个 JSONL、共 128 条记录，每条 rollout 16 条。每条均保留非空原始 `model_output`；重新调用 `extract_progress` 与 fallback recurrence 后，`parsed_progress` 和 `progress` 全部逐条一致。
- RynnValue：8 个 `raw_model_outputs.json`、共 128 个 value-head 值。每条含 16 个有限 value、升序 sampled frame indices、原始 analysis text，以及 `description/match/success` 解析结果。
- Robo-Dopamine：8 个官方 `pred_vllm.json`、共 218 条记录。worker result 精确指向官方文件；`pred` 的 `<score>...</score>` 原文保留，解析出的数值与 `progress` 一致。官方实际输出包含小数百分比（如 `+3.3%`），没有被 wrapper 改写为整数。

## Compatibility changes and checks

迁移后的 [run_baseline.sh](/mnt/hdd/qiuxia/pyr/LF3R/tools/baselines/run_baseline.sh) 使用自身位置推导 LF3R 根目录、加载 `project_env.sh` 并调用 runner；help、dry-run、环境检查和实际 8-rollout 运行均通过，因此不需要改 launcher 路径逻辑。

为适配当前“只看显存余量”的要求，做了以下最小兼容修改：

- [run_lf3r_baseline.py](/mnt/hdd/qiuxia/pyr/LF3R/tools/baselines/run_lf3r_baseline.py:277)：增加 `--vllm-gpu-memory-utilization`，并转发给 ProcVLM/Robo-Dopamine。
- [inference.py](/mnt/hdd/qiuxia/pyr/LF3R/repos/ProcVLM/evqa/inference.py:51)：增加可选 vLLM 显存参数并传入 engine。
- [robo_dopamine_persistent_worker.py](/mnt/hdd/qiuxia/pyr/LF3R/tools/baselines/robo_dopamine_persistent_worker.py:27)：当前唯一 Robo-Dopamine 入口；在单个持久 `GRMInference` 中包装官方模块自己的 `LLM` 符号覆盖默认 0.9，并按 rollout 保存官方 raw 输出。

检查结果：

- `python3 tools/baselines/validate_pipeline.py --check-environments --execute-safe-smoke`：`BASELINE_PIPELINE_VALIDATION_OK`
- `python3 tools/baselines/test_worker_contracts.py`：`BASELINE_WORKER_CONTRACTS_OK`
- 最终统一审计：`BASELINE_REPRESENTATIVE_AUDIT_OK`

未执行全量 baseline 评估；本次仅完成用户指定的 8 条 representative rollout 验证。

