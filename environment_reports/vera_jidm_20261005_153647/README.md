# VERA J-IDM 环境

状态：JIDM_ENV_READY。检查时间：2026-10-05T15:52:40.209163+08:00。

仅配置 J-IDM 所需环境；未安装 `[video]` / `[eval]` 扩展，未下载 WM、IDM 权重或数据，未启动训练、服务器或机器人。官方源码包含完整 VERA，但 Jacobian 训练模块验证没有加载 `vera.video_model`。

## 使用

从 LF3R 根目录开始：

```bash
source ./project_env.sh
test "$PROJECT_ROOT" = "$(pwd -P)"
conda activate "$LF3R_ENV_VERA_JIDM"
```

查看官方 J-IDM 训练配置（不会开始训练）：

```bash
./tools/run_vera_jidm.sh --help
./tools/run_vera_jidm.sh --config-name=config_jacobian_mimicgen_vggt_v3_taskbalanced --help
```

启动器默认选择 PushT J-IDM，默认单卡 `CUDA_VISIBLE_DEVICES=0`，关闭在线 W&B。上面的 MimicGen 参数可覆盖默认配置。用户可自行选择 GPU；环境检查仅用空闲显存决定小型 CUDA 运算是否可运行。

以后准备好数据后，可按官方 TRAINING.md 开始 J-IDM 训练，例如：

```bash
CUDA_VISIBLE_DEVICES=1 ./tools/run_vera_jidm.sh \
    dataset.pusht_zarr_root="$DATASETS/pusht"
```

这条命令会开始训练，本次没有执行。默认打包数据目录是 `$DATASETS/jacobian/pusht_packed`；MimicGen 对应 `$DATASETS/jacobian/mimicgen_packed_v3_megaflow`。模型预训练骨干按官方实现可能首次使用时下载，缓存位于 LF3R/cache；IDM 检查点目录为 `$CHECKPOINTS/vera-jidm`，运行输出位于 `$OUTPUTS/vera-jidm`。

## 环境与来源

- Python 3.11.16；PyTorch 2.7.1+cu128；Torchvision 0.22.1+cu128；Lightning 2.5.1.post0。
- NumPy 1.26.4；Transformers 4.51.3；Zarr 3.0.8；OpenCV 4.11.0.86；rerun-sdk 0.22.1。
- VERA commit：`e9bf1c8033a31be9c42edd6d7348e92c0cfdf9eb`；VGGT commit：`a288dd0f14786c93483e45524328726ab7b1b4ce`。
- 主机：RTX PRO 6000 Blackwell Max-Q，sm_120，驱动 610.57.04；安装前约 156 GiB 磁盘空余。
- [官方 VERA](https://github.com/sizhe-li/VERA)、[官方训练说明](https://github.com/sizhe-li/VERA/blob/main/TRAINING.md)、[PyTorch Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

## 本地适配

`tools/patches/vera-jidm-blackwell-project-cache.patch` 保存全部上游改动：

1. 将 Torch 2.6.0 改为支持本机 Blackwell 的 2.7.1；通过官方 cu128 wheel 安装，不更改系统 CUDA。
2. 固定 VGGT Git 依赖的 commit。
3. 将视频实验的 WAN 注册导入延迟到实际选择 video_generation 时执行，J-IDM 不再要求 WAN 的依赖。
4. 让检查点下载路径支持 `VERA_CKPT_ROOT`，启动器设置为 LF3R/checkpoints/vera-jidm。

`tools/requirements-vera-jidm.txt` 补充上游 idm extra 遗漏的共享依赖。Diffusers 是 eager IDM 注册链使用的共享时间步嵌入库，不代表配置或加载 WM。VGGT 要求 NumPy<2，因此固定兼容的 OpenCV/Zarr/rerun-sdk。删除了可选 decord：其公开 wheel 文件的平台元数据为 Python 3.6，pip/UV 在 Python 3.11 下均报告不兼容。本环境验证官方 packed 数据路径；原始 DROID 视频读取还需另行解决 decord。

导入时可见可选 DFoT 的 `NoTrainLpips` 警告；属于未配置的 WM/DFoT 路径，不影响已经通过检查的 J-IDM。Transformers 的 TRANSFORMERS_CACHE 弃用提示也不影响运行。没有为解决 WM 警告安装额外运行栈。

## 验证及范围

| 检查 | 结果 | 日志 |
| --- | --- | --- |
| J-IDM、数据、Jacobian 训练导入和注册；未加载 WM | PASS | `logs/vera_jidm_registry_20261005_155035.log` |
| 官方 Hydra 训练 CLI --help | PASS | `logs/vera_jidm_cli_20261005_154707.log` |
| pip check | PASS | `logs/vera_jidm_pip_native_check_20261005_154708.log` |
| uv pip check | PASS | `logs/vera_jidm_pip_check_20261005_154708.log` |
| Blackwell 单卡分配与 32x32 matmul | PASS | `logs/vera_jidm_cuda_20261005_154835.log` |

CUDA 检查选择当时空闲显存最多的 GPU 1，峰值张量分配约 8.1 MiB，结果为 32；这是运行时验证，未执行模型推理。完整训练、预训练权重加载、实际数据读取和仿真 rollout 不在本次验证范围。

首次导入发现缺少共享 diffusers，补齐后 J-IDM 导入通过；训练模块发现 eager WAN 导入，按需导入补丁修复后通过。最后的注册器检查脚本曾误写 `ALGORITHM_REGISTRY`（上游实际为 `ALGO_REGISTRY`）；改用官方 `list_algorithms()` 后验证通过。所有失败日志均保留。

完整环境清单：`pip-freeze.txt`、`conda-explicit.txt`；机器与检查日志索引：`manifest.json`。重建入口：`./tools/setup_vera_jidm.sh`，该脚本未重复执行整套安装；安装命令、补丁检查和 Bash 语法均已验证。
