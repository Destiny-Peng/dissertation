# LF3R baseline environment rebuild audit

- Date: 2026-08-26 (Asia/Singapore)
- Project root: `/mnt/hdd/qiuxia/pyr/LF3R`
- Python runtime: CPython 3.10.21 installed by uv under `cache/uv/python`
- Environment manager: uv-managed virtual environments; no `conda` executable is available on this server. The historical `conda_envs/` names are retained as project-local prefixes.
- Scope: environment reconstruction and smoke validation only. No full dataset evaluation, training, or external process termination was performed.

## Checkpoint inventory

All five requested local checkpoints are present and contain `config.json` plus model weight files:

| Checkpoint | Local path | Approx. size |
| --- | --- | ---: |
| OpenVLA LIBERO Spatial | `checkpoints/openvla-7b-finetuned-libero-spatial` | 15G |
| OpenVLA LIBERO-10 | `checkpoints/openvla-7b-finetuned-libero-10` | 15G |
| ProcVLM-2B | `checkpoints/ProcVLM-2B` | 9.2G |
| RynnValue-4B | `checkpoints/RynnValue-4B` | 9.6G |
| Robo-Dopamine GRM 2.0 4B Preview | `checkpoints/Robo-Dopamine-GRM-2.0-4B-Preview` | 9.1G |

Local-only config/tokenizer preflight passed for ProcVLM, RynnValue, and Robo-Dopamine. OpenVLA config parsing passed through the repository-local `prismatic` classes, and a local image/text preprocessing pass produced `input_ids`, `attention_mask`, and `pixel_values` with shape `(1, 6, 224, 224)`. The fine-tuned OpenVLA checkpoint's Transformers metadata points at the base Hub dynamic-code files; the evaluator's repository-local registration is therefore required.

## Environment results

| Component | Environment | Key runtime | Imports / entry point | CUDA tensor smoke | Dependency status |
| --- | --- | --- | --- | --- | --- |
| SAFE | `conda_envs/LF3R-safe` | torch 2.11.0+cu128; Python 3.10.21 | `failure_prob`, OpenCV, pandas imports passed | PASS; RTX PRO 6000 Blackwell | PASS |
| OpenVLA | `conda_envs/LF3R-openvla` | torch 2.11.0+cu128; transformers 4.40.1; TensorFlow 2.15.0; MuJoCo 2.3.7 | core imports and official `run_libero_eval.py --help` passed; local OpenVLA config/processor path passed | PASS; RTX PRO 6000 Blackwell | 3 known legacy metadata conflicts from `openvla` (`torch==2.2.0`, `torchvision==0.17.0`, `torchaudio==2.2.0`) retained for the Blackwell cu128 override |
| Robo-Dopamine | `conda_envs/LF3R-robo-dopamine` | torch 2.8.0+cu128; transformers 4.57.0; vLLM 0.11.0 | inference imports and `examples/inference.py` / `eval/evaluation_grm.py` compilation passed | PASS; RTX PRO 6000 Blackwell | PASS |
| ProcVLM | `repos/ProcVLM/.venv` | torch 2.10.0+cu128; transformers 4.57.3; vLLM 0.18.1; decord 0.6.0 | `evqa.inference --help`, config/tokenizer, and imports passed | PASS; RTX PRO 6000 Blackwell | One upstream decord wheel platform-tag warning; import and video dependency load pass |
| RynnValue | `repos/RynnValue/.venv` | torch 2.11.0+cu128; transformers 4.57.6 | `rynn_infer/inference.py --help`, config/tokenizer, and imports passed | PASS; RTX PRO 6000 Blackwell | PASS |

Every CUDA smoke used a single process and completed a synchronized CUDA tensor operation (`sum=20.0`) with `torch.version.cuda=12.8`, `torch.cuda.device_count()=3`, and device name `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`.

## LIBERO / MuJoCo

- The initial LIBERO headless path failed with MuJoCo 3.12 and `MUJOCO_GL=osmesa` because no system OSMesa library was available.
- MuJoCo was changed to the historical LIBERO-compatible `2.3.7` wheel. `MUJOCO_GL=egl` imports successfully.
- A one-task EGL reset was attempted, but the process was terminated while the three GPUs were already occupied by unrelated `/home/wenkai/.../sfe/bin/python` processes under `nvidia-cuda-mps-server`. No external process was stopped or modified.
- Before using the OpenVLA/LIBERO evaluator, export `MUJOCO_GL=egl` and `MPLBACKEND=Agg`. The reset should be rerun when a GPU is available.

## GPU / model-load gate

At audit time all three GPUs reported 100% utilization, with approximately 22--23 GiB used and 73--75 GiB free, and the active compute processes belonged to the unrelated `sfe` workload. To avoid competing with that workload, this audit did not load the 2B/4B/7B model weights or start a model forward pass. Thus the current-server model-level single-process inference smoke remains pending; the environment-level CUDA execution and local checkpoint/config readiness are verified.

## Repository revisions

- SAFE `b6036ab`
- safe-openvla `300dce2`
- dlimp_openvla `040105d`
- LIBERO `8f1084e`
- ProcVLM `377523a`
- RynnValue `10e0d33`
- Robo-Dopamine `2c714ab`
- FAIL-Detect `b758e55`

## Package freezes

- [SAFE_FREEZE.txt](SAFE_FREEZE.txt)
- [OPENVLA_FREEZE.txt](OPENVLA_FREEZE.txt)
- [ROBODOPAMINE_FREEZE.txt](ROBODOPAMINE_FREEZE.txt)
- [PROCVLM_FREEZE.txt](PROCVLM_FREEZE.txt)
- [RYNNVALUE_FREEZE.txt](RYNNVALUE_FREEZE.txt)
