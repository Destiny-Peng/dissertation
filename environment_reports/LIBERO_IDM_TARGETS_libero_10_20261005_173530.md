# LIBERO → 原始 VERA IDM target 转换

COMPLETE。保持原始 J-IDM 架构、输入格式、七维输出语义及 checkpoint normalization。LIBERO-10 全部500条官方成功轨迹已转换，137590个有效相邻pair。源RGB/HDF5未修改；原始OSC动作保留为replay对照。

转换调用官方 `SE3QuatDeltaAction` 与 `UnifiedDataset._derive_du`：

- 从观测ee_pos / ee_ori构造位姿，ee_ori rotvec转xyzw quaternion。
- `dT = T[k+1] @ inv(T[k])`；前三维为其平移，后三个arm维为相对rotation rotvec。
- 第七维为观测左手指qpos增量，不是binary gripper command。
- 六维SE3乘50，手指增量乘80，再按du_scale=1和原checkpoint action_abs_scale正规化。
- 不重新估计LIBERO normstats、不拟合affine、不裁剪标签。
- pair(obs[k],obs[k+1])关联原始action[k+1]/state[k+1]。首条action因没有存储的pre-action RGB排除。

逐轨迹NPZ同时存储 action_idm [T-1,7] float32、scaled target、physical delta、原OSC action、位姿/手指状态和索引。图像仍从源HDF5读取；[T,V,3,H,W]、agentview/wrist、128x128、uint8/255、原OpenGL方向。

验证：所有轨迹与独立向量化SE3构造及物理反归一化一致；最大normalized target差异 1.86e-09。500条保存标签的shape/dtype/index核验通过，原checkpoint配置复制字节一致，sample RGB布局和原OSC源数据核验通过。

数据：[manifest.json](../datasets/libero_idm_targets/libero_10_20261005_173530/manifest.json)、[读取说明](../datasets/libero_idm_targets/libero_10_20261005_173530/README.md)。入口 `tools/run_libero_idm_target_conversion.sh`；源码 `tools/vera_libero/convert_idm_targets.py`。时间、命令、环境、上游commit、源HDF5/target/config/code hashes和一次已修复配置初始化失败见provenance/manifest及logs。

这是监督target导出，不是完整packed training dataset。原J-IDM采用flow监督，仍需motion-track/flow、原flow normalization和按完整trajectory划分的train/val/test。未启动训练或WM。预测状态增量到LIBERO OSC的controller adapter仍须独立验证。

用户指定的原始IO约束已替代此前诊断报告中的OSC-target建议；实验结果不变。

完成时间：2026-10-05T17:40:30.150788+08:00。
