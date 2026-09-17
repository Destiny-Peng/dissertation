from __future__ import annotations

import unittest
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]


class RobosuiteLogPathContractTest(unittest.TestCase):
    def test_generation_wrappers_route_robosuite_logs_to_workspace(self) -> None:
        for filename in (
            "generate_libero10_natural.sh",
            "generate_libero_spatial_native.sh",
        ):
            script = (TOOL_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn('ROBOSUITE_LOG_DIR="${LOGS}/robosuite"', script)
            self.assertIn('ROBOSUITE_LOG_FILE="${ROBOSUITE_LOG_DIR}/${RUN_NOTE}.log"', script)
            self.assertIn('ROBOSUITE_LOG_PATH="$ROBOSUITE_LOG_FILE"', script)
            self.assertIn('echo "ROBOSUITE_LOG_PATH=$ROBOSUITE_LOG_FILE"', script)

    def test_runner_redirects_only_robosuite_default_file_handler(self) -> None:
        runner = (TOOL_ROOT / "run_openvla_libero10_natural.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("ROBOSUITE_LOG_PATH")', runner)
        self.assertIn('candidate == "/tmp/robosuite.log"', runner)
        self.assertIn("logging.FileHandler = RobosuiteRedirectFileHandler", runner)
        self.assertIn("logging.FileHandler = original_file_handler", runner)
        self.assertIn("evaluator = import_openvla_evaluator()", runner)


if __name__ == "__main__":
    unittest.main()
