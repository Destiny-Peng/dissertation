# Robo-Dopamine 启动显存预留

WebUI 的单条 Run 和批量 Run 默认勾选 **Reserve GPU memory**。仅对新启动的 Robo-Dopamine worker 生效；已运行的 worker 不会补充预留。

**Reservation MiB per GPU** 为 `0` 时，按预留进程初始化 CUDA 后的可用显存及 `vllm_free_memory_fraction` 自动计算。多卡 tensor parallel 使用各卡都能容纳的统一总显存比例。始终保留至少 2048 MiB 未占用空间，并在预留中额外包含 2048 MiB 交接缓冲。

指定值表示每张卡的预留总量，包含交接缓冲。例如 `12288` 约占用 12 GiB，其中约 10 GiB 是 vLLM 总显存预算，另有 2 GiB 交接缓冲。指定值必须大于 2048；显存不足会明确失败，不会等待其他进程退出。CUDA context 本身还会额外占用显存。

CLI 默认不启用，给现有 Robo-Dopamine 命令增加：

```bash
--robo-reserve-gpu-memory --robo-reserve-mib 0
```

或者固定每卡预留 12 GiB：

```bash
--robo-reserve-gpu-memory --robo-reserve-mib 12288
```

Run 启动专用 PyTorch 子进程，实际分配 CUDA tensor 并持续持有，然后启动模型 worker。CPU 导入和引擎启动期间保持预留；每个 vLLM GPU worker 在 `init_device()` 开始时请求释放对应 GPU 的 tensor，随后按预留时保存的预算初始化。取消 Run、runner 退出、预留失败或 worker 失败都会结束本 Run 的预留进程。进程通过本地 Unix socket 和随机 token 交接。

预留要等 PyTorch 导入及 CUDA 初始化完成后才成立，冷缓存下这些步骤仍可能慢。交接不是 CUDA 显存的原子转移，也不是 GPU 独占锁：tensor 释放后，模型加载期间其他进程仍可争用空闲显存。预留能保护启动导入阶段的预算，不能保证整段模型加载都独占 GPU。

`run.json` 保存 `gpu_memory_reservation` 和实际引擎预算；`progress.jsonl` 记录 `gpu_reservation_started`、`gpu_reservation_ready`。Run 日志记录预留导入、预算或失败原因，以及各 GPU 的 `LF3R_GPU_RESERVATION_RELEASED`。恢复 Run 会沿用已保存的配置，也可在恢复旧 Run 时通过 CLI 显式启用。
