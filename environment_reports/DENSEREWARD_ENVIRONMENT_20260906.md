# DenseReward environment and minimal inference validation

- Status: **PASS**
- Validation time: 2026-09-06T18:45:02.581865+08:00
- Model: `densereward/densereward-3frame-thinking`
- Model card: https://huggingface.co/densereward/densereward-3frame-thinking
- Local checkpoint: `/mnt/hdd/qiuxia/pyr/LF3R/checkpoints/densereward-3frame-thinking`
- Checkpoint files: 19 files, including both safetensors shards and `system_prompt.txt`
- Dedicated environment: `conda_envs/LF3R-densereward`
- Python: `3.12.3`
- Inference device: `CUDA_VISIBLE_DEVICES=0`
- Precision: BF16; no quantization

## Official path followed

The validation follows the model card: load `AutoProcessor` and `AutoModelForImageTextToText` with `torch_dtype=torch.bfloat16` and `device_map="auto"`; load the bundled `system_prompt.txt`; construct a system message plus a user message containing exactly three RGB images in chronological order (oldest to current) followed by the task text; call `process_vision_info`; run `model.generate(max_new_tokens=32, do_sample=False)`; decode only generated tokens; parse the `<think>WORD</think>` reason and scalar reward.

The model card specifies exactly three frames and a reward in `[0, 1]`. No baseline runner or analysis pipeline was modified or invoked.

## Minimal example

- Source rollout: `outputs/openvla_libero/lf3r-data-natural-libero10-20260824_210557/libero_10/task0--ep5--succ0.mp4`
- Source video: 520 frames at 224x224, 30 FPS
- Selected source frame indices: `[517, 518, 519]`
- Input order: `frame-01` (oldest), `frame-02`, `frame-03` (current)
- Task: `put both the alphabet soup and the tomato sauce in the basket`
- System prompt: `/mnt/hdd/qiuxia/pyr/LF3R/checkpoints/densereward-3frame-thinking/system_prompt.txt`
- System prompt SHA-256: `7987b38ea84a3df789e6a9eb6fb551f805849d3662d6278bda2f5ffcb4d0b5bd`
- Input tokens: `750`
- Generated tokens: `12`

Raw model output:

```text
<think>
correct
</think>

0.521
```

Parsed result:

- reason: `correct`
- reward: `0.521`
- finite and within range: `true`
- reward clipped: `False`

## Exact command

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  conda_envs/LF3R-densereward/bin/python tools/densereward_minimal_inference.py \
  --model-path checkpoints/densereward-3frame-thinking \
  --task "put both the alphabet soup and the tomato sauce in the basket" \
  --frame outputs/densereward_smoke_20260906/frames/frame-01.png \
  --frame outputs/densereward_smoke_20260906/frames/frame-02.png \
  --frame outputs/densereward_smoke_20260906/frames/frame-03.png \
  --output outputs/densereward_smoke_20260906/result.json
```

Artifacts:

- Result: `outputs/densereward_smoke_20260906/result.json`
- Log: `logs/densereward_smoke_20260906.log`
- Frames: `outputs/densereward_smoke_20260906/frames/`

## Environment package freeze

```text
accelerate==1.13.0
annotated-doc==0.0.5
anyio==4.15.1
av==18.1.0
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
filelock==3.32.3
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.30.0
idna==3.19
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.2
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
packaging==26.3
pillow==12.3.0
psutil==7.2.2
pygments==2.21.0
pyyaml==6.0.3
qwen-vl-utils==0.0.14
regex==2026.9.3
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
setuptools==78.1.0
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.22.2
torch==2.8.0+cu128
torchvision==0.23.0+cu128
tqdm==4.70.0
transformers==5.2.0
triton==3.4.0
typer==0.27.2
typer-slim==0.24.0
typing-extensions==4.16.0
urllib3==2.7.0
```

A non-fatal Transformers warning reported that `torch_dtype` is deprecated in favor of `dtype`; the official model-card path still completed successfully with BF16.
