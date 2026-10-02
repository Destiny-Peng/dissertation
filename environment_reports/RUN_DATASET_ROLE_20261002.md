# Run dataset_role 筛选与 manifest 分类

核对时间：2026-10-02T14:43:04.266796+08:00；HEAD：`db2a383`（工作区修改）。

当前聚合 manifest 共 1528 条，12 个角色；无缺失 dataset_role。分类来自现有字段，没有重写源 manifest。

| dataset_role | 数量 |
| --- | ---: |
| `droid_failure_subset` | 20 |
| `droid_matched_failure_success` | 11 |
| `libero_10` | 477 |
| `libero_10_official_success` | 121 |
| `libero_spatial` | 41 |
| `realrobot` | 586 |
| `realrobot_failrecovery` | 151 |
| `realrobot_tube` | 51 |
| `reboot_matched_failure_success` | 15 |
| `reboot_recovery_subset` | 15 |
| `robovad_anomaly_subset` | 20 |
| `robovad_matched_failure_success` | 20 |

`realrobot` 包含 eggplant 314 条、ring 272 条。`libero_10` 是自然策略轨迹，官方成功演示单列为 `libero_10_official_success`。

Run 下拉选项及共享 scope 后端均按 dataset_role 精确匹配；保留 All。范围索引和 worker 范围在角色筛选后计算；runner 使用明确 rollout IDs，避免 task_suite 和 partition 再次扩大范围。共享 scope 的结果查询和 analysis 范围同步采用角色语义。

验证：三个 HTTP 测试通过，覆盖自然/官方同 suite 分离、首条官方演示时索引仍选择自然轨迹、manifest 自定义 role。真实 manifest 的 Node 控制器验证全部 12 个角色的选项、数量和前后端匹配一致。

命令：
```bash
python3 -m unittest tools.lf3r_annotator.tests.test_server.ServerTest.test_baseline_scope_separates_dataset_roles_in_same_suite tools.lf3r_annotator.tests.test_server.ServerTest.test_baseline_run_discovery_and_batch_endpoint tools.lf3r_annotator.tests.test_server.ServerTest.test_baseline_batch_accepts_manifest_defined_dataset_role
node tmp/check_dataset_role_frontend.cjs
git diff --check
```

测试使用系统 Python3 与 Node24；HTTP 测试仅运行 fake runner，无模型推理。

WebUI 更新会话：`lf3r-annotator-dataset-role`，通过 `bash tools/lf3r_annotator/run_server.sh 8765` 启动。启动进程在触觉索引读取阶段等待磁盘；尚未声称在线接口验证通过。

服务完成启动，在线验证通过：`/api/manifests` 返回全部 12 个 dataset_roles，合计 1528；自然 LIBERO-10=477，官方演示=121；新版本静态 scope 控制器已提供。
