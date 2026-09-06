# FAIL-Detect Dependency Report

Repository commit: `b758e55f7c0c988188f2e4876ffc03ae8a3c30ed`.

The official environment pins Python 3.9, PyTorch 1.12.1, CUDA Toolkit 11.6, torchvision 0.13.1, and PyTorch3D 0.7.0, plus older NumPy, numba, Robomimic, Robosuite, and MuJoCo bindings. This legacy GPU environment was deliberately not created because it predates Blackwell support. No system CUDA, driver, or OS library was changed.

The schema-compatible native sample is `datasets/FAIL-Detect/robomimic/square/ph/minimal_native_sample.hdf5`. It contains one 16-step demonstration with two 84x84 RGB streams, low-dimensional robot observations, 7D actions, states, rewards, dones, and Robomimic `env_args` metadata. Size: 36,959 bytes. SHA-256: `a4e922a6ebe146fec39562fb1f7383a4a8f6b130a4f8f05dbaaab584bd537e17`.

