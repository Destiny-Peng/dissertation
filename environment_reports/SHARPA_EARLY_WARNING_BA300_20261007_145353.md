# Controlled Deform GRU BA300 repeat

Completed 2026-10-07T14:57:25.769466+08:00

User corrected criterion: validation BA (not loss). Only max30→300 and BA patience8→50 changed; five horizons seed42, original code training body reused. User explicitly authorized parallel GPU execution, overriding default sequential guideline. GPU0:H0, GPU1:H8, GPU2:H15/H30/H45. Environment repos/ProcVLM/.venv, no installation or download.

```bash
source ./project_env.sh
bash tools/run_early_warning_ba300.sh outputs/sharpa_early_warning_ba300/20261007_145353
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -u -m sharpa_tactile.early_warning_ba300_report --output outputs/sharpa_early_warning_ba300/20261007_145353
```

本轮确认：五个H的validation/test逐帧预测与旧seed42完全一致（最大绝对差0）；训练重叠前缀的train BCE和validation BA也完全一致。H0/8/15/30均在epoch52停止并恢复epoch2，H45在epoch55停止并恢复epoch5。延长训练上限和BA patience没有改变最终结果，因此该seed下旧结果可精确复现；此前三阶段实验的差异不能归因于只延长训练轮数。它仍改变了architecture、loss、checkpoint criterion等多个因素，尚不能进一步单独归因。

Source scripts/weights/annotations unchanged; new helper scripts and original-code snapshot stored with output. Prefix loss/BA audit, GT-count comparison, automatic stopping and saved-prediction replay PASS. Source/weekly README, CSV and figures copied without alteration. Relevant code identity is recorded by SHA256 in each shard manifest and code_snapshot; no commit created.
