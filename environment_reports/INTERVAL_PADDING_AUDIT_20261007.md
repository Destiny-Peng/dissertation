# Interval padding audit

Command (GPU2):
```bash
source ./project_env.sh
PYTHONPATH="$PROJECT_ROOT/tools" "$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python" -m sharpa_tactile.audit_interval_padding --output outputs/sharpa_interval_padding_audit/20261007_161501 --device cuda:2
```

Source: outputs/sharpa_failure_relabel/20261006_165000. No training rerun, no test. All15 checkpoint hashes recorded; all archived probabilities reproduced within2e-6. Correct variants have zero class changes; padding input gradients0. Length models fit only train.

Three command failures were syntax, cuDNN backward called in eval mode, and relative-path report publication. Retries stopped after third failure per AGENTS.md. The third occurred after all numerical outputs and output README were written. Only CPU publication of existing artifacts followed; no fourth GPU audit. Code issues corrected; original labels and checkpoints untouched. Weekly artifact copies SHA verified.
