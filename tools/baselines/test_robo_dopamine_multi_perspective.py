#!/usr/bin/env python3
"""Unit tests for native-grid Robo-Dopamine perspective fusion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robo_dopamine_persistent_worker import infer_rollout
from robo_dopamine_multi_perspective import (
    FUSED_EVAL_MODE,
    PERSPECTIVE_MODES,
    fuse_prediction_files,
    normalize_eval_modes,
    resolve_eval_modes,
    summarize_incremental_noise,
)


def _rows(values: list[float]) -> list[dict[str, object]]:
    return [
        {
            "id": f"step-af_{frame:06d}",
            "image": ["unused"] * 5 + [f"frame_{frame:06d}.png"],
            "progress": value,
            "hop": value,
        }
        for frame, value in zip((4, 8, 12), values)
    ]


class MultiPerspectiveTests(unittest.TestCase):
    def test_worker_reuses_one_model_for_three_modes_and_writes_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="robo-worker-multi-") as temporary:
            root = Path(temporary)
            model_path = root / "checkpoint"
            model_path.mkdir()
            goal = root / "blank_goal.png"
            goal.touch()
            output_dir = root / "raw" / "rollout"
            calls: list[tuple[object, str]] = []

            class FakeModel:
                def run_pipeline(self, *, out_root, eval_mode, **kwargs):
                    calls.append((self, eval_mode))
                    official = Path(out_root) / f"official_{eval_mode}"
                    official.mkdir(parents=True, exist_ok=True)
                    rows = _rows({
                        "incremental": [0.1, 0.2, 0.3],
                        "forward": [0.2, 0.3, 0.4],
                        "backward": [0.3, 0.4, 0.5],
                    }[eval_mode])
                    prediction = official / "pred_vllm.json"
                    prediction.write_text(json.dumps(rows), encoding="utf-8")
                    return str(official)

            args = type("Args", (), {
                "repo": root,
                "model_path": model_path,
                "goal_image": goal,
                "frame_interval": 10,
                "batch_size": 1,
                "eval_mode": "forward",
                "eval_modes": list(PERSPECTIVE_MODES),
                "render_video": False,
            })()
            result = infer_rollout(
                {
                    "rollout_id": "rollout",
                    "video_path": str(root / "video.mp4"),
                    "task": "test task",
                    "raw_output_dir": str(output_dir),
                    "goal_image": str(goal),
                },
                args,
                FakeModel(),
            )
            self.assertEqual([mode for _, mode in calls], list(PERSPECTIVE_MODES))
            self.assertEqual(len({id(model) for model, _ in calls}), 1)
            worker_result = json.loads((output_dir / "worker_result.json").read_text())
            self.assertTrue(worker_result["multi_perspective"])
            self.assertEqual(set(worker_result["perspective_outputs"]), set(PERSPECTIVE_MODES))
            fused = json.loads(Path(result["fused_model_output"]).read_text())
            for row, expected in zip(fused, [0.2, 0.3, 0.4]):
                self.assertAlmostEqual(row["progress"], expected)

    def test_modes_are_normalized_in_official_order(self) -> None:
        self.assertEqual(
            normalize_eval_modes("backward,incremental,forward"),
            list(PERSPECTIVE_MODES),
        )

    def test_fused_wrapper_mode_expands_but_legacy_mode_does_not(self) -> None:
        self.assertEqual(resolve_eval_modes(FUSED_EVAL_MODE), list(PERSPECTIVE_MODES))
        self.assertEqual(resolve_eval_modes("forward"), ["forward"])
        self.assertEqual(
            resolve_eval_modes(FUSED_EVAL_MODE, ["forward"]),
            ["forward"],
        )

    def test_fusion_uses_matching_native_indices_and_mean_progress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="robo-fusion-") as temporary:
            root = Path(temporary)
            paths = {}
            values = {
                "incremental": [0.2, 0.4, 0.6],
                "forward": [0.3, 0.5, 0.7],
                "backward": [0.4, 0.6, 0.8],
            }
            for mode in PERSPECTIVE_MODES:
                path = root / f"{mode}.json"
                path.write_text(json.dumps(_rows(values[mode])), encoding="utf-8")
                paths[mode] = path
            output = root / "fused.json"
            summary = fuse_prediction_files(paths, output)
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_indices"], [4, 8, 12])
            self.assertEqual([row["frame_index"] for row in rows], [4, 8, 12])
            for actual, expected in zip(
                [row["progress"] for row in rows], [0.3, 0.5, 0.7]
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                [row["hop"] for row in rows], [0.3, 0.2, 0.2]
            ):
                self.assertAlmostEqual(actual, expected)

    def test_grid_mismatch_is_rejected_without_interpolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="robo-fusion-grid-") as temporary:
            root = Path(temporary)
            paths = {}
            for mode in PERSPECTIVE_MODES:
                path = root / f"{mode}.json"
                rows = _rows([0.1, 0.2, 0.3])
                if mode == "backward":
                    rows[-1]["image"] = ["unused"] * 5 + ["frame_0013.png"]
                path.write_text(json.dumps(rows), encoding="utf-8")
                paths[mode] = path
            with self.assertRaisesRegex(ValueError, "grids do not match"):
                fuse_prediction_files(paths, root / "fused.json")

    def test_incremental_noise_summary_is_finite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="robo-noise-") as temporary:
            path = Path(temporary) / "incremental.json"
            path.write_text(json.dumps(_rows([0.1, -0.2, 0.3])), encoding="utf-8")
            result = summarize_incremental_noise(path)
            self.assertEqual(result["samples"], 3)
            self.assertGreater(result["sign_change_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

