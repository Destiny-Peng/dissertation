"""Offline import and registry smoke; no model construction or weight download."""
import sys

import torch
import vera.idm
import vera.datasets
from vera.experiments.jacobian_learning import JacobianLearningExperiment
from vera.idm.registry import list_algorithms
from vera.idm.jacobian.models.registry import list_models

assert "image_jacobian" in list_algorithms()
assert not any(name.startswith("vera.video_model") for name in sys.modules)
print("PASS J-IDM, datasets, JacobianLearningExperiment imports")
print("Algorithms:", list_algorithms())
print("Models:", list_models())
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("PASS no vera.video_model modules imported")
