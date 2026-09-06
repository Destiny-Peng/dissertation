from __future__ import annotations

import csv
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - the OpenVLA environment supplies these.
    np = None
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = PROJECT_ROOT / "tools" / "lf3r_annotator"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

if torch is not None:
    from safe_feature_logging import (
        ACTION_COLUMNS,
        _hidden_states_to_numpy,
        extract_safe_hidden_states,
        postprocess_episode,
    )


@unittest.skipIf(torch is None or np is None, "numpy and torch are required")
class SafeFeatureLoggingTest(unittest.TestCase):
    def test_extraction_matches_official_last_layer_last_position(self) -> None:
        generated = []
        for token_index in range(7):
            generated.append(
                tuple(
                    torch.full(
                        (1, 2, 4096),
                        float(token_index * 100 + layer_index),
                        dtype=torch.float32,
                    )
                    for layer_index in range(3)
                )
            )

        extracted = extract_safe_hidden_states({"hidden_states": tuple(generated)})

        self.assertEqual(tuple(extracted.shape), (7, 4096))
        self.assertTrue(torch.all(extracted[0] == 2.0))
        self.assertTrue(torch.all(extracted[-1] == 602.0))

    def test_bfloat16_conversion_does_not_require_numpy_view_of_torch_tensor(self) -> None:
        array, source_dtype = _hidden_states_to_numpy(
            torch.ones((2, 7, 4096), dtype=torch.bfloat16)
        )
        self.assertEqual(source_dtype, "bfloat16")
        self.assertEqual(array.shape, (2, 7, 4096))
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(np.isfinite(array).all())

    def test_postprocessor_merges_official_pkl_and_csv_without_mutating_pkl(self) -> None:
        step_count = 3
        hidden_states = torch.arange(
            step_count * 7 * 4096, dtype=torch.float32
        ).reshape(step_count, 7, 4096)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp4_path = root / "task0--ep0--succ1.mp4"
            mp4_path.write_bytes(b"fake-replay")
            pkl_path = mp4_path.with_suffix(".pkl")
            with pkl_path.open("wb") as handle:
                pickle.dump(
                    {
                        "hidden_states": hidden_states,
                        "task_suite_name": "libero_10",
                        "task_id": 0,
                        "task_description": "test task",
                        "eposide_idx": 0,
                        "episode_success": True,
                        "mp4_path": str(mp4_path),
                    },
                    handle,
                )
            original_pkl = pkl_path.read_bytes()
            csv_path = mp4_path.with_suffix(".csv")
            fieldnames = ["action/timestep", *ACTION_COLUMNS]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(step_count):
                    writer.writerow(
                        {
                            "action/timestep": 10 + index,
                            **{
                                column: value
                                for column, value in zip(
                                    ACTION_COLUMNS,
                                    [0.1 * index, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                                )
                            },
                        }
                    )

            npz_path, metadata_path = postprocess_episode(
                pkl_path=pkl_path,
                checkpoint=PROJECT_ROOT / "checkpoints/openvla-7b-finetuned-libero-10",
                project_root=PROJECT_ROOT,
                task_suite_name="libero_10",
            )

            self.assertEqual(pkl_path.read_bytes(), original_pkl)
            with np.load(npz_path) as arrays:
                self.assertEqual(arrays["hidden_states"].shape, (3, 7, 4096))
                self.assertEqual(arrays["actions"].shape, (3, 7))
                self.assertEqual(arrays["environment_timestep"].tolist(), [10, 11, 12])
                self.assertEqual(arrays["frame_index"].tolist(), [0, 1, 2])
                self.assertEqual(int(arrays["episode_success"]), 1)
                self.assertTrue(np.isfinite(arrays["hidden_states"]).all())
                self.assertTrue(np.isfinite(arrays["actions"]).all())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["representation"]["shape"], [3, 7, 4096])
            self.assertEqual(
                metadata["representation"]["safe_default_input_shape"],
                [3, 4096],
            )
            self.assertEqual(
                metadata["representation"]["selection"],
                'generated_outputs["hidden_states"][token][-1][0, -1, :]',
            )
            self.assertEqual(
                metadata["alignment"]["count_validation"]["hidden_state_steps"],
                3,
            )
            self.assertTrue(metadata["compatibility"]["official_pkl_unchanged"])


if __name__ == "__main__":
    unittest.main()
