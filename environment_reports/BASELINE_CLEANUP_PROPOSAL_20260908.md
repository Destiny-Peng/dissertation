# Baseline 结果清理清单（2026-09-08）

## 结论

本次盘点没有发现仍在运行的 baseline、analysis 或 rollout generation 任务。
当前结果没有直接移动或删除：Analysis snapshot 和历史报告保存了原始 run 路径，直接移动有效目录会破坏 provenance。下面将结果分为：

1. 当前 Analysis / Review 必须保留的 canonical 结果；
2. 已验证有效、但属于 smoke / sanity / 历史诊断的结果；
3. 明确没有可用 raw output、可以清理的失败或空结果；
4. 有 raw output 但不是当前 canonical source、建议暂时保留的历史结果。

## 当前 canonical 结果（不要删除）

当前高采样 primary analysis 使用以下目录：

| Method | Run |
|---|---|
| SAFE | `outputs/baselines/full_136/safe_20260826_212113_540452` |
| ProcVLM | `outputs/baselines/full_136_01/procvlm_20260827_190222_580731` |
| RynnValue | `outputs/baselines/web_runs/rynnvalue-batch-fe9e05c5ba54/rynnvalue_20260828_222438_749771` |
| RynnValue | `outputs/baselines/web_runs/rynnvalue-batch-cc8deb3f2e09/rynnvalue_20260829_221918_087826` |
| Robo-Dopamine | `outputs/baselines/full_136_01/robo_dopamine_20260827_200551_687444` |

RynnValue 的两个 web run 逻辑合并后覆盖 124/125 条，之前已确认其中 43 条结果有效，因此不能清理。

最新完整 Robo-Dopamine 结果也必须保留：

`outputs/baselines/web_runs/robo_dopamine-batch-326e663e25bc/robo_dopamine_20260908_105724_234057`

它是 166/166 complete、0 failed 的 full-instruction 结果。

## 已验证有效的历史 / 诊断结果（建议保留）

这些目录不是当前 primary analysis 的 canonical source，但包含可用 raw output 或有效的 sanity 结果：

- `outputs/baselines/full_136/rynnvalue_20260826_212344_287240` — 125/125，旧低采样 RynnValue。
- `outputs/baselines/full_136/procvlm_20260826_212239_165575` — 54/125，部分有效 raw output。
- `outputs/baselines/full_136/robo_dopamine_20260826_212444_097971` — 117/125，部分有效 raw output。
- `outputs/baselines/full_136_01/procvlm_20260827_190222_580731` — 125/125。
- `outputs/baselines/persistent_procvlm_representative_20260827/procvlm_20260827_180045_610051` — 8/8。
- `outputs/baselines/persistent_robo_dopamine_representative_20260827/robo_dopamine_20260827_193829_137208` — 8/8。
- `outputs/baselines/representative_20260826/procvlm/procvlm_20260826_194404_483223` — 1/1。
- `outputs/baselines/representative_20260826/procvlm/procvlm_20260826_194717_717199` — 8/8。
- `outputs/baselines/representative_20260826/robo_dopamine/robo_dopamine_20260826_202750_573074` — 1/1。
- `outputs/baselines/representative_20260826/robo_dopamine/robo_dopamine_20260826_203106_485078` — 8/8。
- `outputs/baselines/representative_20260826/rynnvalue/rynnvalue_20260826_195300_710951` — 8/8。
- `outputs/baselines/representative_20260826/safe/safe_20260826_193421_272938` — 8/8。
- `outputs/baselines/free_memory_smoke_20260827/procvlm_20260827_160033_373567` — 2/2。
- `outputs/baselines/free_memory_smoke_20260827/robo_dopamine_20260827_160349_222723` — 2/2。
- `outputs/baselines/rynnvalue_allframes_smoke_20260828_retry/rynnvalue_20260828_142933_320206` — retry smoke 成功。
- `outputs/baselines/test/rynnvalue_20260829_203137_616977` — 1/1，唯一完成的 test run。
- `outputs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201426_242626` — 2/2，多视角 sanity 成功。
- `outputs/baselines/robo_dopamine_interval_sanity_20260905/interval_sweep_20260905_204010_369904` — interval sweep 完成。
- `outputs/baselines/densereward_smoke_20260906/densereward_20260906_213813_833577` — 1/1 DenseReward smoke。
- `outputs/baselines/web_runs/densereward-batch-5fde64e07be2/densereward_20260906_222656_492155` — 125/125 primary natural。
- `outputs/baselines/web_runs/densereward-batch-639c778ccee3/densereward_20260906_215556_418318` — 41/41 reference natural。
- `outputs/baselines/web_runs/safe-batch-eeff930ca0f9/safe_20260904_220414_841929` — 41/41 reference natural。

## 建议清理组 A：明确无效的结果目录和对应日志

以下项目没有可用 raw output，或已经被成功 retry / 新结果取代。它们可以在确认后删除。日志也没有有效推理结果，只保留失败原因；如果希望保留故障追踪记录，可以只删除 output 目录、保留日志。

### RynnValue 未产出 raw 的测试 / OOM

- `outputs/baselines/test/rynnvalue_20260829_160611_494884`
- `logs/baselines/test/rynnvalue_20260829_160611_494884.log`
- `outputs/baselines/test/rynnvalue_20260829_162749_705136`
- `logs/baselines/test/rynnvalue_20260829_162749_705136.log`
- `outputs/baselines/full_136_01/rynnvalue_20260828_145343_220340`
- `logs/baselines/full_136_01/rynnvalue_20260828_145343_220340.log`
- `outputs/baselines/rynnvalue_allframes_smoke_20260828/rynnvalue_20260828_142635_413547`
- `logs/baselines/rynnvalue_allframes_smoke_20260828/rynnvalue_20260828_142635_413547.log`

其中 `full_136_01` 是 CUDA OOM；前两次 test 没有 raw output 且状态停留在 running；all-frames 初次 smoke 已由 `..._retry` 成功结果取代。

### 失败的 ProcVLM representative 尝试

- `outputs/baselines/representative_20260826/procvlm/procvlm_20260826_193835_107659`
- `logs/baselines/representative_20260826/procvlm/procvlm_20260826_193835_107659.log`

该尝试失败且没有 raw output；同目录下的 `194404` 和 `194717` 是有效结果，不能一起删除。

### Robo-Dopamine web 失败 / 空运行

以下 5 个 web run 都没有 raw output：

- `outputs/baselines/web_runs/robo_dopamine-batch-125ad1e94d0e`
- `outputs/baselines/web_runs/robo_dopamine-batch-3439f46731cf`
- `outputs/baselines/web_runs/robo_dopamine-batch-4401419b151e`
- `outputs/baselines/web_runs/robo_dopamine-batch-5e71e961ed4e`
- `outputs/baselines/web_runs/robo_dopamine-batch-91ab856ccb17`

对应日志如下：

- `logs/baselines/web_runs/robo_dopamine-batch-125ad1e94d0e.log`
- `logs/baselines/web_runs/robo_dopamine_20260905_223922_249763.log`
- `logs/baselines/web_runs/robo_dopamine-batch-3439f46731cf.log`
- `logs/baselines/web_runs/robo_dopamine_20260905_223119_789311.log`
- `logs/baselines/web_runs/robo_dopamine-batch-4401419b151e.log`
- `logs/baselines/web_runs/robo_dopamine_20260905_222344_204124.log`
- `logs/baselines/web_runs/robo_dopamine-batch-5e71e961ed4e.log`
- `logs/baselines/web_runs/robo_dopamine_20260908_105201_068100.log`
- `logs/baselines/web_runs/robo_dopamine-batch-91ab856ccb17.log`
- `logs/baselines/web_runs/robo_dopamine_20260905_222204_395603.log`

失败原因分别是显存门禁不足或 worker 的 `jobs.jsonl` 初始化失败；最新成功的 `batch-326e663e25bc` 不在此清单中。

### 空的 Robo-Dopamine interval sweep 尝试

- `outputs/baselines/robo_dopamine_interval_sanity_20260905/interval_sweep_20260905_203929_567279`

状态仍是 `initializing` 且没有 raw output；同目录下的 `204010_369904` 是有效完成的 sweep，不能删除。

## 建议清理组 B：可选的流程验证文件

这些不是 baseline 结果，删除不会影响 Review，但会减少环境验证的历史记录：

- `outputs/baselines/dryrun_20260826`
- `logs/baselines/dryrun_20260826`
- `outputs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201228_433672`
- `outputs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201245_640238`
- `logs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201228_433672.log`
- `logs/baselines/robo_dopamine_multi_sanity_20260905/robo_dopamine_20260905_201245_640238.log`

其中第一个 Robo-Dopamine sanity 是 dry-run，第二个是 engine failure；已完成的 `201426_242626` 需要保留。

`outputs/baselines/dense_sampling_smoke_20260827` 及其日志虽然有失败记录，但仍被 `environment_reports/BASELINE_DENSE_SAMPLING_SMOKE_20260827.md` 用作环境诊断依据，建议暂时保留。


## 第二轮复筛：重复但有效的结果

这次进一步按“完成的 rollout ID 集合 + 实际参数 + raw 内容”复查，而不是只按目录名判断。结论如下。

### 建议加入清理组 A 的明确重复项

#### RynnValue 单 rollout test

- `outputs/baselines/test/rynnvalue_20260829_203137_616977`
- `logs/baselines/test/rynnvalue_20260829_203137_616977.log`

它只包含 `libero_10-task00-ep000-natural-07acb763b6`，该 rollout 已经在保留的
`outputs/baselines/web_runs/rynnvalue-batch-fe9e05c5ba54/rynnvalue_20260828_222438_749771`
中完成。两者都是 `num_frames=16`、`num_steps=105`、`evaluation_interval=4`，只是 test 使用 `batch_size=2`，web aggregate 使用 `batch_size=1`，输出数值有轻微差异。

网页候选按完成时间选择，保留这个 test 会使 Review 优先读到 test 输出，而不是 43 条 aggregate 结果中的 canonical 输出。因此该 test 应删除，保留 web aggregate。

#### DenseReward 单 rollout smoke

- `outputs/baselines/densereward_smoke_20260906/densereward_20260906_213813_833577`
- `logs/baselines/densereward_smoke_20260906/densereward_20260906_213813_833577.log`

它只覆盖 `libero_10-task00-ep000-natural-07acb763b6`，该 ID 已包含在完整的 125 条 primary run：

`outputs/baselines/web_runs/densereward-batch-5fde64e07be2/densereward_20260906_222656_492155`

如果不再需要 smoke 调试记录，这一对是明确可删除的重复项。

### 同一 rollout 集合，但不是重复结果（默认保留）

以下项目虽然 rollout ID 有重叠，但实际采样或推理模式不同，不能按“重复文件”处理：

- `representative_20260826/procvlm/...194717` 与 `persistent_procvlm_representative_20260827/...180045`：相同 8 个 ID，但 raw 文件逐个不相同，分别代表普通入口和 persistent worker。
- `representative_20260826/robo_dopamine/...203106` 与 `persistent_robo_dopamine_representative_20260827/...193829`：相同 8 个 ID，但分别使用 `frame_interval=15` 和 `frame_interval=4`。
- `free_memory_smoke_20260827/robo_dopamine_...160349` 与 `robo_dopamine_multi_sanity_20260905/...201426`：相同 2 个 ID，但前者是 forward、interval=4，后者是 multi-perspective、interval=10。
- `full_136/procvlm_...212239` 与 `full_136_01/procvlm_...190222`：前者只有 54 条完成结果，且是旧的逐 rollout 入口；后者是 125 条完整结果。
- `full_136/robo_dopamine_...212444` 与 `full_136_01/robo_dopamine_...200551`：前者是旧的 forward、interval=15，后者是 high-resolution、interval=2。
- `full_136/rynnvalue_...212344` 与两个 RynnValue web aggregate：前者是旧的低采样 125 条结果，不能替代高采样 aggregate，也不能用于填补当前 124/125 的缺失项。

如果只保留当前 canonical 结果，可以另行删除这些历史 / 诊断目录；但这会使旧环境报告中的链接失效，并丢失不同采样策略的对照数据，所以本轮不把它们列为明确重复。

## 当前建议

建议清理组 A，并把上面“明确重复项”中的 RynnValue test 和 DenseReward smoke 一并加入。组 B 由你决定。不要删除 canonical 和“历史 / 诊断结果”两节中的目录。

本次仍没有执行 `rm`，原因是清理项同时包含 output 和故障日志，而日志是否保留属于数据保留策略。确认后可以按这份清单只处理明确路径，不会使用宽泛的 glob 或递归删除项目根目录。
