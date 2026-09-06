# FAIL-Detect Blackwell Porting Notes

Status: `FAILDETECT_GPU_PORT_REQUIRED`.

Use an isolated project-local environment with Python 3.10/3.11 and Blackwell-capable PyTorch. Rebuild or replace PyTorch3D against the selected PyTorch/CUDA ABI, then update NumPy/numba, zarr/numcodecs, Robomimic, Robosuite, and MuJoCo as a tested set. Likely hotspots are Gym APIs, `free-mujoco-py`, Hydra/OmegaConf behavior, zarr v2 APIs, `torch.load` defaults, and compiled PyTorch3D operators.

Validate imports and the minimal native sample on CPU first. Only then run one resource-gated GPU smoke test. Checkpoint acquisition, policy training, and full Robomimic datasets remain out of scope.
