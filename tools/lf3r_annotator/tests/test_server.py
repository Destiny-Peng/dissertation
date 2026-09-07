from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import LF3RApplication, make_handler


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        video = self.root / "outputs" / "sample.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"0123456789abcdef")
        self.rollout = {
            "schema_version": 1,
            "id": "sample-rollout",
            "task_suite": "libero_10",
            "task_id": 0,
            "episode_index": 0,
            "task_description": "test task",
            "ground_truth_outcome": "failure",
            "source_kind": "natural_policy",
            "analysis_partition": "natural_observation",
            "dataset_role": "primary_natural",
            "video_path": "outputs/sample.mp4",
            "total_frames": 10,
            "fps": 5.0,
            "first_environment_timestep": 10,
            "injection": None,
        }
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(self.rollout) + "\n", encoding="utf-8")
        analysis_python = (
            Path(__file__).resolve().parents[3]
            / "conda_envs"
            / "LF3R-ananlyse"
            / "bin"
            / "python"
        )
        self.app = LF3RApplication(
            self.root,
            manifest,
            self.root / "annotations",
            analysis_python=analysis_python,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path: str, data: dict | None = None, headers: dict | None = None):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(
            self.base + path,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST" if data is not None else "GET",
        )
        return urllib.request.urlopen(request, timeout=3)

    def put_request(self, path: str, data: dict):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        return urllib.request.urlopen(request, timeout=3)


    def seed_baseline_outputs(self) -> None:
        baseline_root = self.root / "outputs" / "baselines"
        run_metadata = {
            "schema_version": 1,
            "status": "complete",
            "created_at": "2026-08-27T00:00:00+00:00",
            "completed_at": "2026-08-27T00:00:01+00:00",
            "selected_rollouts": 1,
            "completed_jobs": 1,
            "failed_jobs": 0,
        }

        safe_root = baseline_root / "safe_test"
        safe_raw = safe_root / "raw" / self.rollout["id"]
        safe_raw.mkdir(parents=True)
        (safe_root / "run.json").write_text(
            json.dumps({**run_metadata, "baseline": "safe"}) + "\n", encoding="utf-8"
        )
        (safe_raw / "safe_features.csv").write_text(
            "action_timestep,max_token_prob,avg_token_prob\n10,0.9,0.8\n12,0.7,0.6\n",
            encoding="utf-8",
        )
        (safe_root / "jobs.jsonl").write_text(json.dumps({"rollout_id": self.rollout["id"], "status": "complete"}) + "\n", encoding="utf-8")

        proc_root = baseline_root / "procvlm_test"
        proc_raw = proc_root / "raw" / self.rollout["id"]
        proc_raw.mkdir(parents=True)
        (proc_root / "run.json").write_text(
            json.dumps({**run_metadata, "baseline": "procvlm"}) + "\n", encoding="utf-8"
        )
        (proc_raw / "procvlm_raw.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"frame_index": 0, "progress": 0.1, "model_output": "start", "reasoning": "observe"}),
                    json.dumps({"frame_index": 99, "progress": 0.9, "model_output": "late", "reasoning": "act"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (proc_root / "jobs.jsonl").write_text(json.dumps({"rollout_id": self.rollout["id"], "status": "complete"}) + "\n", encoding="utf-8")

        rynn_root = baseline_root / "rynnvalue_test"
        rynn_raw = rynn_root / "raw" / self.rollout["id"] / "sample"
        rynn_raw.mkdir(parents=True)
        (rynn_root / "run.json").write_text(
            json.dumps({**run_metadata, "baseline": "rynnvalue"}) + "\n", encoding="utf-8"
        )
        (rynn_raw / "raw_model_outputs.json").write_text(
            json.dumps(
                {
                    "values": [0.2, 0.3],
                    "sampled_indices": [1, 3],
                    "analysis_text": "looks stable",
                    "parsed_analysis": {"score": 0.2},
                }
            ),
            encoding="utf-8",
        )
        (rynn_root / "jobs.jsonl").write_text(json.dumps({"rollout_id": self.rollout["id"], "status": "complete"}) + "\n", encoding="utf-8")

        robo_root = baseline_root / "robo_dopamine_test"
        robo_raw = robo_root / "raw" / self.rollout["id"]
        robo_raw.mkdir(parents=True)
        prediction_path = robo_raw / "pred.json"
        prediction_path.write_text(
            json.dumps(
                [
                    {
                        "image": ["", "", "", "", "", "frame_000002.png"],
                        "progress": 0.4,
                        "hop": 2,
                        "pred": "<score>+3.3%</score>",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (robo_raw / "worker_result.json").write_text(
            json.dumps({"raw_model_output": str(prediction_path)}),
            encoding="utf-8",
        )
        (robo_root / "run.json").write_text(
            json.dumps({**run_metadata, "baseline": "robo_dopamine"}) + "\n", encoding="utf-8"
        )
        (robo_root / "jobs.jsonl").write_text(json.dumps({"rollout_id": self.rollout["id"], "status": "complete"}) + "\n", encoding="utf-8")

        dense_root = baseline_root / "densereward_test"
        dense_raw = dense_root / "raw" / self.rollout["id"]
        dense_raw.mkdir(parents=True)
        (dense_root / "run.json").write_text(
            json.dumps({**run_metadata, "baseline": "densereward"}) + "\n", encoding="utf-8"
        )
        (dense_raw / "densereward_raw.jsonl").write_text(
            json.dumps({
                "frame_index": 2,
                "sampled_frame_indices": [0, 1, 2],
                "reward": 0.521,
                "reason": "correct",
                "raw_text": "<think>correct</think>\n\n0.521",
            }) + "\n",
            encoding="utf-8",
        )
        (dense_raw / "worker_result.json").write_text(
            json.dumps({"baseline": "densereward", "raw_model_output": str(dense_raw / "densereward_raw.jsonl")}),
            encoding="utf-8",
        )
        (dense_root / "jobs.jsonl").write_text(json.dumps({"rollout_id": self.rollout["id"], "status": "complete"}) + "\n", encoding="utf-8")

    def test_baseline_evaluation_reads_all_output_types(self) -> None:
        self.seed_baseline_outputs()
        with self.request("/api/baselines/sample-rollout") as response:
            evaluation = json.load(response)["evaluation"]
        self.assertEqual(
            set(evaluation["available_methods"]),
            {"safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"},
        )
        methods = evaluation["methods"]
        self.assertEqual(methods["safe"]["sample_count"], 2)
        self.assertEqual(methods["procvlm"]["samples"][0]["model_output"], "start")
        self.assertEqual(methods["procvlm"]["validation"]["status"], "warning")
        self.assertEqual(methods["rynnvalue"]["samples"][0]["analysis_text"], "looks stable")
        self.assertEqual(methods["robo_dopamine"]["samples"][0]["pred"], "<score>+3.3%</score>")
        self.assertEqual(methods["densereward"]["samples"][0]["signals"]["reward"], 0.521)


    def test_robo_dopamine_web_command_defaults_to_fused(self) -> None:
        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# test runner\n", encoding="utf-8")
        command = self.app.baselines._baseline_command(
            "robo_dopamine",
            "all",
            "0",
            0.80,
            self.root / "outputs" / "baselines" / "web_runs",
            {},
        )
        self.assertEqual(command[command.index("--robo-eval-mode") + 1], "fused")
        legacy = self.app.baselines._baseline_command(
            "robo_dopamine",
            "all",
            "0",
            0.80,
            self.root / "outputs" / "baselines" / "web_runs",
            {"robo_eval_mode": "forward"},
        )
        self.assertEqual(legacy[legacy.index("--robo-eval-mode") + 1], "forward")

    def test_baseline_run_is_bounded_to_one_rollout(self) -> None:
        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("# test runner\n", encoding="utf-8")
        with self.request(
            "/api/baselines/run/sample-rollout",
            {"baseline": "safe", "gpu": "0", "memory_utilization": 0.80},
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["rollout_id"], "sample-rollout")
        self.assertEqual(job["memory_utilization"], 0.80)
        self.assertEqual(job["memory_scope"], "free_gpu_memory")
        self.assertIn("--rollout-id", job["command"])
        partition_index = job["command"].index("--partition")
        self.assertEqual(job["command"][partition_index + 1], "all")
        final = job
        for _ in range(50):
            with self.request("/api/baseline-jobs/" + job["job_id"]) as response:
                final = json.load(response)["job"]
            if final["status"] not in {"queued", "running"}:
                break
            time.sleep(0.02)
        self.assertEqual(final["status"], "complete")

    def test_invalid_baseline_run_request_is_rejected(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                "/api/baselines/run/sample-rollout",
                {"baseline": "not-a-baseline", "gpu": "0", "memory_utilization": 0.65},
            )
        self.assertEqual(caught.exception.code, 400)

    def test_health_manifest_and_range_video(self) -> None:
        with self.request("/api/health") as response:
            health = json.load(response)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["rollouts"], 1)
        with self.request("/api/rollouts") as response:
            records = json.load(response)["rollouts"]
        self.assertEqual(records[0]["analysis_partition"], "natural_observation")
        self.assertEqual(records[0]["annotation_status"], "unreviewed")
        with self.request(
            "/api/videos/sample-rollout", headers={"Range": "bytes=2-5"}
        ) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/16")
            self.assertEqual(response.read(), b"2345")

    def test_annotation_is_saved_and_reloaded(self) -> None:
        payload = {
            "annotator": "reviewer-a",
            "review_status": "complete",
            "outcome_label": "failure",
            "failure_type": "grasp_failure",
            "confidence": 4,
            "causal_onset_frame": 2,
            "observable_onset_frame": 4,
            "terminal_failure_frame": 7,
            "notes": "Visible slip.",
        }
        with self.request("/api/annotations/sample-rollout", payload) as response:
            saved = json.load(response)["annotation"]
        self.assertEqual(saved["rollout_id"], "sample-rollout")
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["observable_onset_frame"], 4)
        self.assertEqual(len(saved["failure_events"]), 1)
        self.assertEqual(saved["failure_events"][0]["observable_onset_frame"], 4)
        with self.request("/api/annotations/sample-rollout") as response:
            loaded = json.load(response)["annotation"]
        self.assertEqual(loaded, saved)
        record_path = self.root / "annotations" / "records" / "sample-rollout.json"
        self.assertTrue(record_path.is_file())
        self.assertTrue(list((self.root / "annotations" / "events").glob("*.jsonl")))

    def test_recovered_success_saves_multiple_failure_events(self) -> None:
        payload = {
            "annotator": "reviewer-a",
            "review_status": "complete",
            "outcome_label": "recovered_success",
            "failure_type": "grasp_failure",
            "confidence": 5,
            "failure_events": [
                {
                    "failure_type": "grasp_failure",
                    "causal_onset_frame": 1,
                    "observable_onset_frame": 2,
                    "terminal_failure_frame": None,
                    "recovery_frame": 3,
                    "notes": "First grasp recovered.",
                },
                {
                    "failure_type": "placement_failure",
                    "causal_onset_frame": 4,
                    "observable_onset_frame": 5,
                    "terminal_failure_frame": None,
                    "recovery_frame": 8,
                    "notes": "Second attempt recovered.",
                },
            ],
            "notes": "Succeeded after two failed attempts.",
        }
        with self.request("/api/annotations/sample-rollout", payload) as response:
            saved = json.load(response)["annotation"]
        self.assertEqual(saved["outcome_label"], "recovered_success")
        self.assertEqual(len(saved["failure_events"]), 2)
        self.assertEqual(saved["failure_events"][1]["recovery_frame"], 8)
        self.assertEqual(saved["causal_onset_frame"], 1)

    def test_recovered_success_rejects_terminal_event(self) -> None:
        payload = {
            "annotator": "reviewer-a",
            "outcome_label": "recovered_success",
            "failure_type": "other",
            "failure_events": [{"observable_onset_frame": 2, "terminal_failure_frame": 5}],
        }
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/annotations/sample-rollout", payload)
        self.assertEqual(caught.exception.code, 400)

    def test_failure_event_list_is_not_limited_to_five(self) -> None:
        event = {
            "failure_type": "grasp_failure",
            "causal_onset_frame": 1,
            "observable_onset_frame": 2,
            "terminal_failure_frame": None,
            "recovery_frame": 3,
            "notes": "",
        }
        payload = {
            "annotator": "reviewer-a",
            "outcome_label": "recovered_success",
            "failure_type": "grasp_failure",
            "failure_events": [dict(event) for _ in range(12)],
        }
        with self.request("/api/annotations/sample-rollout", payload) as response:
            saved = json.load(response)["annotation"]
        self.assertEqual(len(saved["failure_events"]), 12)

    def test_invalid_onset_order_is_rejected(self) -> None:
        payload = {
            "annotator": "reviewer-a",
            "review_status": "complete",
            "outcome_label": "failure",
            "failure_type": "other",
            "confidence": 3,
            "causal_onset_frame": 6,
            "observable_onset_frame": 4,
            "terminal_failure_frame": 7,
            "notes": "",
        }
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/annotations/sample-rollout", payload)
        self.assertEqual(caught.exception.code, 400)

    def test_manifest_cannot_escape_project_root(self) -> None:
        app = LF3RApplication(
            self.root,
            self.root / "manifest.jsonl",
            self.root / "annotations-two",
        )
        with self.assertRaises(ValueError):
            app.resolve_project_file("../outside.mp4", ".mp4")


    def test_settings_defaults_roundtrip_and_validation(self) -> None:
        with self.request("/api/settings") as response:
            payload = json.load(response)
        defaults = payload["settings"]
        self.assertEqual(defaults["background_color"], "#0b0d10")
        self.assertEqual(defaults["font_scale"], 1.0)
        self.assertEqual(defaults["density"], "comfortable")
        self.assertIsNone(payload["updated_at"])

        settings = dict(defaults)
        settings.update({
            "background_color": "#ABCDEF",
            "accent_color": "#123456",
            "font_scale": 1.15,
            "density": "spacious",
        })
        with self.put_request("/api/settings", settings) as response:
            saved = json.load(response)
        self.assertEqual(saved["settings"]["background_color"], "#abcdef")
        self.assertEqual(saved["settings"]["font_scale"], 1.15)
        self.assertTrue(saved["updated_at"])
        with self.request("/api/settings") as response:
            loaded = json.load(response)
        self.assertEqual(loaded["settings"], saved["settings"])
        self.assertTrue((self.root / "config" / "lf3r_annotator.json").is_file())

        invalid_payloads = [
            dict(settings, background_color="#fff"),
            dict(settings, font_scale=0.70),
            dict(settings, font_scale=1.07),
            dict(settings, review_font_scale=0.80),
            dict(settings, analysis_font_scale=1.35),
            dict(settings, control_font_scale=1.07),
            dict(settings, density="dense"),
            dict(settings, unknown_field=True),
        ]
        for invalid in invalid_payloads:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.put_request("/api/settings", invalid)
            self.assertEqual(caught.exception.code, 400)

        (self.root / "config" / "lf3r_annotator.json").write_text("{\"unknown\": true}\n", encoding="utf-8")
        with self.request("/api/settings") as response:
            fallback = json.load(response)
        self.assertEqual(fallback["settings"], defaults)
        self.assertIsNone(fallback["updated_at"])

    def seed_analysis_runs(self, missing_method: str | None = None, include_extra: bool = False) -> dict[str, str]:
        roots = {}
        baseline_root = self.root / "outputs" / "baselines" / "analysis_inputs"
        extra_id = "extra-rollout"
        for method in ("safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"):
            root = baseline_root / method
            root.mkdir(parents=True, exist_ok=True)
            ids = [] if method == missing_method else [self.rollout["id"]]
            if include_extra:
                ids.append(extra_id)
            metadata = {
                "schema_version": 1,
                "status": "complete",
                "baseline": method,
                "selected_rollouts": len(ids) if ids else 1,
                "completed_jobs": len(ids),
                "failed_jobs": 0,
                "created_at": "2026-08-28T00:00:00+00:00",
                "completed_at": "2026-08-28T00:00:01+00:00",
            }
            (root / "run.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            (root / "jobs.jsonl").write_text(
                "".join(json.dumps({"rollout_id": rollout_id, "status": "complete"}) + "\n" for rollout_id in ids),
                encoding="utf-8",
            )
            roots[method] = str(root.relative_to(self.root))
        return roots

    def install_fake_temporal_analyzer(self) -> None:
        script = self.root / "tools" / "analyze_baseline_temporal_signals.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            """import argparse\nimport json\nfrom pathlib import Path\n\nparser = argparse.ArgumentParser()\nparser.add_argument('--output-dir', type=Path, required=True)\nparser.add_argument('--selection', type=Path, required=True)\nargs = parser.parse_known_args()[0]\nargs.output_dir.mkdir(parents=True, exist_ok=True)\nselection = json.loads(args.selection.read_text())\n(args.output_dir / 'metadata.json').write_text(json.dumps({'counts': {'rollouts': len(selection['selection'])}, 'selection': str(args.selection)}))\n(args.output_dir / 'event_metrics.jsonl').write_text('')\nfor name in ('method_coverage.csv', 'summary_by_method_signal_outcome.csv', 'summary_by_method_outcome.csv', 'onset_signal_statistics.csv', 'clean_background_summary.csv'):\n    (args.output_dir / name).write_text('method\\n')\nprint('fake temporal analysis complete')\n""",
            encoding="utf-8",
        )

    def install_fake_rollout_generator(self, exit_code: int = 0) -> None:
        script = self.root / "tools" / "lf3r_annotator" / "generate_libero10_natural.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            f"""#!/bin/sh
set -eu
GPU="$1"
START="$2"
END="$3"
TRIALS="$4"
SEED="$5"
NOTE="$6"
SAFE_FEATURES="${{7:-}}"
ROOT="$(pwd)"
OUT="$ROOT/outputs/openvla_libero/$NOTE/libero_10"
mkdir -p "$OUT"
printf '%s\n' "$GPU" "$START" "$END" "$TRIALS" "$SEED" "$NOTE" "$SAFE_FEATURES" > "$ROOT/fake_generator_args"
echo "fake generator gpu=$GPU start=$START end=$END trials=$TRIALS seed=$SEED run_note=$NOTE"
if [ "{exit_code}" -ne 0 ]; then
    echo "fake generator exit {exit_code}"
    exit "{exit_code}"
fi
TASK="$START"
while [ "$TASK" -le "$END" ]; do
    EP=0
    while [ "$EP" -lt "$TRIALS" ]; do
        : > "$OUT/task$TASK--ep$EP--succ1.mp4"
        EP=$((EP + 1))
    done
    TASK=$((TASK + 1))
done
printf '\n' >> "$ROOT/manifest.jsonl"
""",
            encoding="utf-8",
        )
        script.chmod(0o755)

    def install_fake_spatial_rollout_generator(self, exit_code: int = 0) -> None:
        script = self.app.rollout_jobs.scripts["libero_spatial"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            f"""#!/bin/sh
set -eu
GPU="$1"
START="$2"
END="$3"
TRIALS="$4"
SEED="$5"
NOTE="$6"
SAFE_FEATURES="${{7:-}}"
ROOT="$(pwd)"
OUT="$ROOT/outputs/openvla_libero_spatial_native/$NOTE/libero_spatial"
mkdir -p "$OUT"
printf '%s\\n' "$GPU" "$START" "$END" "$TRIALS" "$SEED" "$NOTE" "$SAFE_FEATURES" > "$ROOT/fake_spatial_generator_args"
echo "fake spatial generator gpu=$GPU start=$START end=$END trials=$TRIALS seed=$SEED run_note=$NOTE"
if [ "{exit_code}" -ne 0 ]; then
    echo "fake spatial generator exit {exit_code}"
    exit "{exit_code}"
fi
TASK="$START"
while [ "$TASK" -le "$END" ]; do
    EP=0
    while [ "$EP" -lt "$TRIALS" ]; do
        : > "$OUT/task$TASK--ep$EP--succ1.mp4"
        EP=$((EP + 1))
    done
    TASK=$((TASK + 1))
done
printf '\\n' >> "$ROOT/manifest.jsonl"
""",
            encoding="utf-8",
        )
        script.chmod(0o755)

    def wait_for_job(self, endpoint: str, job_id: str, key: str = "job") -> dict:
        result = {}
        for _ in range(100):
            with self.request(endpoint + "/" + job_id) as response:
                result = json.load(response)[key]
            if result["status"] not in {"queued", "running"}:
                return result
            time.sleep(0.02)
        self.fail("job did not finish: " + job_id)

    def test_baseline_run_discovery_and_batch_endpoint(self) -> None:
        self.seed_baseline_outputs()
        with self.request("/api/baselines/runs?scope=primary_natural") as response:
            payload = json.load(response)
        self.assertEqual(payload["scope"], "primary_natural")
        self.assertEqual(len(payload["runs"]), 5)
        self.assertTrue(all(run["compatible"] for run in payload["runs"]))
        with self.request("/api/baselines/runs?scope=controlled_analysis") as response:
            controlled = json.load(response)
        self.assertTrue(all(not run["compatible"] for run in controlled["runs"]))

        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# test runner\n", encoding="utf-8")
        with self.request(
            "/api/baselines/run-batch",
            {
                "baseline": "safe",
                "scope": "primary_natural",
                "gpu": "0,1",
                "memory_utilization": 0.65,
                "start_index": 0,
                "limit": 1,
                "options": {"render_video": True, "dry_run": True},
            },
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["scope"], "primary_natural")
        self.assertEqual(job["selected_rollouts"], 1)
        self.assertEqual(job["memory_scope"], "free_gpu_memory")
        self.assertIn("--partition", job["command"])
        partition_index = job["command"].index("--partition")
        self.assertEqual(job["command"][partition_index + 1], "natural_observation")
        self.assertIn("--dataset-role", job["command"])
        self.assertIn("--dry-run", job["command"])
        final = self.wait_for_job("/api/baseline-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")
        with self.request("/api/baseline-jobs/" + job["job_id"] + "/log?tail=10") as response:
            log = json.load(response)["log"]
        self.assertEqual(log["job_id"], job["job_id"])

    def test_batch_validation_and_baseline_concurrency(self) -> None:
        invalid_payloads = [
            {"baseline": "safe", "scope": "primary_natural", "gpu": "0,x"},
            {"baseline": "safe", "scope": "primary_natural", "gpu": "0", "memory_utilization": True},
            {"baseline": "safe", "scope": "primary_natural", "gpu": "0", "options": {"model_path": "bad"}},
            {"baseline": "safe", "scope": "not-a-scope", "gpu": "0"},
            {"baseline": "safe", "scope": "primary_natural", "gpu": "0", "unexpected": 1},
        ]
        for payload in invalid_payloads:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/baselines/run-batch", payload)
            self.assertEqual(caught.exception.code, 400)

        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# independent fake runner\n", encoding="utf-8")
        jobs = []
        for method in ("safe", "procvlm"):
            with self.request(
                "/api/baselines/run-batch",
                {
                    "baseline": method,
                    "scope": "primary_natural",
                    "gpu": "0",
                    "memory_utilization": 0.80,
                    "options": {"dry_run": True},
                },
            ) as response:
                self.assertEqual(response.status, 202)
                jobs.append(json.load(response)["job"])
        self.assertNotEqual(jobs[0]["job_id"], jobs[1]["job_id"])
        self.assertEqual(jobs[0]["gpu"], jobs[1]["gpu"])
        self.assertTrue(all(job["persistent"] for job in jobs))
        for job in jobs:
            final = self.wait_for_job("/api/baseline-jobs", job["job_id"])
            self.assertEqual(final["status"], "complete")

        self.app.job_coordinator.acquire("manifest-job", "manifest_writer")
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(
                    "/api/baselines/run-batch",
                    {"baseline": "safe", "scope": "primary_natural", "gpu": "0"},
                )
            self.assertEqual(caught.exception.code, 409)
        finally:
            self.app.job_coordinator.release("manifest-job")

    def test_worker_assignments_are_forwarded_for_every_baseline(self) -> None:
        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# fake all-method worker runner\n", encoding="utf-8")
        for method in ("safe", "procvlm", "rynnvalue", "robo_dopamine", "densereward"):
            with self.request(
                "/api/baselines/run-batch",
                {
                    "baseline": method,
                    "scope": "primary_natural",
                    "gpu": "0",
                    "start_index": 0,
                    "end_index": 1,
                    "parallel_workers": 1,
                    "workers": [{"gpu": "0", "start_index": 0, "end_index": 1}],
                    "options": {"dry_run": True},
                },
            ) as response:
                self.assertEqual(response.status, 202)
                job = json.load(response)["job"]
            self.assertEqual(job["parallel_workers"], 1)
            self.assertEqual(job["worker_assignments"][0]["gpu"], "0")
            self.assertIn("--worker-spec", job["command"])
            self.assertIn("0:0:1", job["command"])
            final = self.wait_for_job("/api/baseline-jobs", job["job_id"])
            self.assertEqual(final["status"], "complete")

    def test_rynnvalue_batch_worker_ranges_and_same_gpu_are_forwarded(self) -> None:
        plan = self.app.baselines._worker_plan(
            "rynnvalue",
            0,
            10,
            "0",
            2,
            [
                {"gpu": "0", "start_index": 0, "end_index": 6},
                {"gpu": "0", "start_index": 5, "end_index": 10},
            ],
        )
        self.assertEqual(plan["parallel_workers"], 2)
        self.assertEqual(plan["overlaps"], [5])
        self.assertEqual(plan["gaps"], [])
        self.assertEqual(plan["unique_selected_rollouts"], 10)
        self.assertEqual([item["gpu"] for item in plan["worker_assignments"]], ["0", "0"])

        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# fake Rynn runner\n", encoding="utf-8")
        with self.request(
            "/api/baselines/run-batch",
            {
                "baseline": "rynnvalue",
                "scope": "primary_natural",
                "gpu": "0",
                "start_index": 0,
                "end_index": 1,
                "workers": [{"gpu": "0", "start_index": 0, "end_index": 1}],
                "options": {"rynn_batch_size": 5000, "dry_run": True},
            },
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["parallel_workers"], 1)
        self.assertEqual(job["worker_assignments"][0]["gpu"], "0")
        self.assertIn("--end-index", job["command"])
        self.assertIn("--worker-spec", job["command"])
        self.assertIn("0:0:1", job["command"])
        self.assertEqual(job["options"]["rynn_batch_size"], 5000)
        final = self.wait_for_job("/api/baseline-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")

    def test_analysis_run_creates_snapshot_from_superset_runs(self) -> None:
        self.install_fake_temporal_analyzer()
        roots = self.seed_analysis_runs(include_extra=True)
        with self.request(
            "/api/analysis/run",
            {
                "scope": "primary_natural",
                "runs": roots,
                "pre_window_frames": 60,
                "post_window_frames": 60,
                "background_stride_frames": 30,
                "output_label": "web_test",
            },
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["scope"], "primary_natural")
        self.assertEqual(job["selected_rollouts"], 1)
        selection_path = self.root / job["selection_path"]
        self.assertTrue(selection_path.is_file())
        self.assertEqual(json.loads(selection_path.read_text())["selection"], [{"id": self.rollout["id"]}])
        final = self.wait_for_job("/api/analysis-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")
        output_dir = self.root / final["output_dir"]
        self.assertTrue((output_dir / "metadata.json").is_file())
        with self.request("/api/analysis-jobs/" + job["job_id"] + "/log?tail=20") as response:
            log = json.load(response)["log"]
        self.assertIn("fake temporal analysis complete", log["text"])
        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["source"]["selection_count"], 1)

    def test_analysis_run_rejects_missing_ids_and_paths(self) -> None:
        self.install_fake_temporal_analyzer()
        roots = self.seed_analysis_runs(missing_method="safe")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/analysis/run", {"scope": "primary_natural", "runs": roots})
        self.assertEqual(caught.exception.code, 400)
        roots = self.seed_analysis_runs()
        roots["safe"] = "../outside-run"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/analysis/run", {"scope": "primary_natural", "runs": roots})
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/analysis/run", {"scope": "controlled_analysis", "runs": roots})
        self.assertEqual(caught.exception.code, 400)

    def test_analysis_snapshot_selects_latest_and_compacts_events(self) -> None:
        analysis_root = self.root / "outputs" / "baseline_signal_analysis"
        selection = self.root / "selection.jsonl"
        selection.write_text(self.rollout["id"] + "\n", encoding="utf-8")
        manifest_hash = hashlib.sha256((self.root / "manifest.jsonl").read_bytes()).hexdigest()

        def write_snapshot(name: str, event: dict) -> Path:
            directory = analysis_root / name
            directory.mkdir(parents=True)
            metadata = {
                "schema_version": 1,
                "selection": str(selection),
                "manifest_sha256": manifest_hash,
                "counts": {"rollouts": 1},
                "pre_window_frames": 2,
                "post_window_frames": 4,
                "background_stride_frames": 3,
                "frame_coordinate": "video_frame_index",
                "native_sampling_preserved": True,
                "methods": ["safe"],
            }
            (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (directory / "event_metrics.jsonl").write_text(
                json.dumps(event) + "\n" + json.dumps({"is_background": True, "rollout_id": self.rollout["id"]}) + "\n",
                encoding="utf-8",
            )
            (directory / "method_coverage.csv").write_text(
                "method,selected_rollouts,available_rollouts,missing_rollouts,run_status,run_completed_jobs,run_failed_jobs\n"
                "safe,1,1,0,complete,1,0\n",
                encoding="utf-8",
            )
            (directory / "summary_by_method_signal_outcome.csv").write_text(
                "method,signal,outcome_group,n_events,normalized_response_magnitude_median,post_event_persistence_fraction_median,recovery_fraction_toward_baseline_median\n"
                "safe,max_token_prob,recovered_success,1,0.75,0.5,1.0\n",
                encoding="utf-8",
            )
            (directory / "summary_by_method_outcome.csv").write_text(
                "method,outcome_group,n_events\n safe,recovered_success,1\n",
                encoding="utf-8",
            )
            (directory / "onset_signal_statistics.csv").write_text(
                "method,signal,outcome_group,n_onset_events,n_exceeding_q95,exceeding_q95_fraction,direction_consistency_fraction_all\n"
                "safe,max_token_prob,recovered_success,1,1,1.0,1.0\n",
                encoding="utf-8",
            )
            (directory / "clean_background_summary.csv").write_text(
                "method,signal,n_pseudo_events,normalized_response_magnitude_q95\n"
                "safe,max_token_prob,10,0.9\n",
                encoding="utf-8",
            )
            return directory

        event = {
            "rollout_id": self.rollout["id"],
            "event_index": 0,
            "event_type": "annotated_event",
            "outcome_group": "recovered_success",
            "failure_type": "grasp_failure",
            "is_background": False,
            "event_frame": 3,
            "recovery_frame": 7,
            "terminal_failure_frame": None,
            "next_event_frame": None,
            "recovery_offset_frames": 4,
            "method": "safe",
            "signal": "max_token_prob",
            "normalized_response_magnitude": 0.75,
            "normalized_failure_oriented_response": 0.75,
            "post_event_persistence_fraction": 0.5,
            "post_event_persistence_median_oriented": 0.75,
            "recovery_rebound_normalized": 0.5,
            "recovery_fraction_toward_baseline": 1.0,
            "clean_background_response_q95": 0.9,
            "clean_background_response_percentile": 0.95,
            "pre_sample_frames": [1, 2, 3],
            "post_sample_frames": [3, 4, 5],
        }
        older = write_snapshot("older", event)
        os.utime(older / "metadata.json", (time.time() - 20, time.time() - 20))
        newer = write_snapshot("newer", event)
        os.utime(newer / "metadata.json", (time.time(), time.time()))

        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["source"]["directory"], "outputs/baseline_signal_analysis/newer")
        self.assertEqual(analysis["freshness"]["snapshot_rollout_count"], 1)
        self.assertTrue(analysis["freshness"]["manifest_matches"])
        self.assertEqual(len(analysis["event_metrics"]), 1)
        compact = analysis["event_metrics"][0]
        self.assertEqual(compact["event_frame"], 3)
        self.assertIsInstance(compact["normalized_response_magnitude"], float)
        self.assertEqual(compact["task_suite"], "libero_10")
        self.assertEqual(compact["task_id"], 0)
        self.assertNotIn("pre_sample_frames", compact)
        self.assertEqual(analysis["method_coverage"][0]["available_rollouts"], 1)

    def test_analysis_snapshot_exposes_change_point_artifacts(self) -> None:
        analysis_root = self.root / "outputs" / "baseline_signal_analysis"
        directory = analysis_root / "changepoint_test"
        directory.mkdir(parents=True)
        selection = self.root / "selection.json"
        selection.write_text(json.dumps({"selection": [{"id": self.rollout["id"]}]}), encoding="utf-8")
        manifest_hash = hashlib.sha256((self.root / "manifest.jsonl").read_bytes()).hexdigest()
        metadata = {
            "schema_version": 1,
            "analysis": "lf3r_baseline_change_points",
            "selection": str(selection),
            "manifest_sha256": manifest_hash,
            "generated_at": "2026-08-30T00:00:00+00:00",
            "counts": {"rollouts": 1, "observable_events": 1},
            "methods": ["safe"],
            "features": ["level"],
            "local_scales_frames": [8],
            "thresholds": ["q95"],
            "tolerances_frames": [8],
            "native_sampling_preserved": True,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "changepoint_event_metrics.jsonl").write_text(
            json.dumps({
                "method": "safe",
                "signal": "max_token_prob",
                "scale_frames": 8,
                "feature": "level",
                "threshold": "q95",
                "rollout_id": self.rollout["id"],
                "event_index": 0,
                "outcome_group": "terminal_failure",
                "failure_type": "grasp_failure",
                "observable_onset_frame": 4,
                "peak_distance_frames": 2,
                "first_exceedance_absolute_error_frames": 2,
                "hit": True,
                "native_samples_only": True,
                "frame_values": [1, 2, 3],
            }) + "\n",
            encoding="utf-8",
        )
        files = {
            "method_coverage.csv": "method,selected_rollouts,available_rollouts,missing_rollouts" + chr(10) + "safe,1,1,0" + chr(10),
            "changepoint_summary.csv": (
                "method,signal,scale_frames,feature,threshold,outcome_group,n_events,recall" + chr(10)
                + "safe,max_token_prob,16,level,q95,all_events,1,1.0" + chr(10)
            ),
            "changepoint_reference_summary.csv": (
                "method,signal,scale_frames,feature,threshold_q95" + chr(10)
                + "safe,max_token_prob,16,level,1.0" + chr(10)
            ),
            "changepoint_by_failure_type.csv": (
                "method,signal,scale_frames,feature,threshold,failure_type,n_events,recall" + chr(10)
                + "safe,max_token_prob,16,level,q95,grasp_failure,1,1.0" + chr(10)
            ),
            "changepoint_scales.csv": (
                "method,signal,scale_frames,feature,threshold_q95" + chr(10)
                + "safe,max_token_prob,16,level,1.0" + chr(10)
            ),
        }

        for filename, content in files.items():
            (directory / filename).write_text(content, encoding="utf-8")
        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertTrue(analysis["available"])
        self.assertFalse(analysis["temporal_available"])
        self.assertTrue(analysis["change_point_available"])
        self.assertEqual(analysis["primary_analysis_type"], "change_point")
        self.assertEqual(
            analysis["primary_analysis_source"]["directory"],
            "outputs/baseline_signal_analysis/changepoint_test",
        )
        self.assertTrue(analysis["legacy_temporal_available"] is False)
        change_point = analysis["change_point"]
        self.assertTrue(change_point["localization_available"])
        self.assertEqual(len(change_point["localization_summary"]), 1)
        self.assertEqual(len(change_point["localization_thresholds"]), 1)
        self.assertEqual(len(change_point["localization_by_failure_type"]), 1)
        self.assertEqual(
            change_point["source"]["directory"],
            "outputs/baseline_signal_analysis/changepoint_test",
        )
        self.assertEqual(len(change_point["summary"]), 1)
        self.assertEqual(len(change_point["event_metrics"]), 1)
        self.assertNotIn("frame_values", change_point["event_metrics"][0])
        self.assertEqual(change_point["event_metrics"][0]["task_id"], 0)

    def test_analysis_snapshot_exposes_event_triggered_artifacts(self) -> None:
        analysis_root = self.root / "outputs" / "baseline_signal_analysis"
        directory = analysis_root / "event_triggered_test"
        directory.mkdir(parents=True)
        selection = self.root / "selection.json"
        selection.write_text(
            json.dumps({"selection": [{"id": self.rollout["id"]}]}),
            encoding="utf-8",
        )
        manifest_hash = hashlib.sha256(
            (self.root / "manifest.jsonl").read_bytes()
        ).hexdigest()
        metadata = {
            "schema_version": 1,
            "analysis": "lf3r_event_triggered_signal_analysis",
            "selection": str(selection),
            "manifest_sha256": manifest_hash,
            "generated_at": "2026-08-31T00:00:00+00:00",
            "methods": ["safe"],
            "signals_by_method": {"safe": ["max_token_prob"]},
            "parameters": {
                "pre_window_frames": 2,
                "post_window_frames": 2,
                "native_sampling_preserved": True,
            },
            "event_group_counts": {
                "terminal_failure": 1,
                "recovered_success": 0,
                "uncertain": 0,
            },
            "native_sampling": {
                "safe": {
                    "max_token_prob": {
                        "interval_median_frames": 1.0,
                    }
                }
            },
            "counts": {
                "rollouts": 1,
                "observable_events": 1,
                "summary_rows": 2,
            },
            "plots": [],
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        files = {
            "event_triggered_curves.csv": chr(10).join([
                "method,signal,comparison_group,group,relative_frame,raw_q25,raw_median,raw_q75,normalized_q25,normalized_median,normalized_q75,n_samples,n_trajectories",
                "safe,max_token_prob,terminal_failure,terminal_failure,-1,0.1,0.2,0.3,-0.2,0.0,0.2,1,1",
                "safe,max_token_prob,terminal_failure,matched_clean_success,-1,0.1,0.1,0.1,-0.1,0.0,0.1,1,1",
            ]) + chr(10),
            "event_triggered_change_scores.csv": chr(10).join([
                "method,signal,comparison_group,group,scale_frames,relative_frame,change_q25,change_median,change_q75,score_q25,score_median,score_q75,n_samples,n_trajectories",
                "safe,max_token_prob,terminal_failure,terminal_failure,8,-1,0.1,0.2,0.3,0.1,0.2,0.3,1,1",
                "safe,max_token_prob,terminal_failure,matched_clean_success,8,-1,0.1,0.1,0.1,0.1,0.1,0.1,1,1",
            ]) + chr(10),
            "event_triggered_separation.csv": chr(10).join([
                "method,signal,event_group,representation,scale_frames,relative_frame,signed_separation,separation_z",
                "safe,max_token_prob,terminal_failure,normalized_signal,,-1,0.0,0.0",
            ]) + chr(10),
            "event_triggered_summary.csv": chr(10).join([
                "method,signal,event_group,representation,scale_frames,n_events,n_controls,strongest_relative_frame,strongest_phase,strongest_signed_separation,strongest_separation_z,typical_lag_frames,before_fraction,at_onset_fraction,after_fraction",
                "safe,max_token_prob,terminal_failure,normalized_signal,,1,1,-1,before,0.1,0.2,-1,1.0,0.0,0.0",
                "safe,max_token_prob,terminal_failure,local_change_score,8,1,1,-1,before,0.1,0.2,-1,1.0,0.0,0.0",
            ]) + chr(10),
            "event_triggered_peak_events.csv": chr(10).join([
                "method,signal,event_group,representation,scale_frames,event_id,rollout_id,event_index,peak_relative_frame,peak_phase,peak_separation_z,failure_type,task_id,control_rollout_id",
                "safe,max_token_prob,terminal_failure,normalized_signal,,event-0,sample-rollout,0,-1,before,0.2,grasp_failure,0,clean-rollout",
            ]) + chr(10),
            "event_triggered_controls.csv": chr(10).join([
                "method,signal,event_id,event_rollout_id,event_group,status,matched_rollout_id,match_level",
                "safe,max_token_prob,event-0,sample-rollout,terminal_failure,matched,clean-rollout,exact_task",
            ]) + chr(10),
            "method_coverage.csv": chr(10).join([
                "method,signal,selected_rollouts,available_method_rollouts,missing_method_rollouts,missing_method_rollout_ids,event_bearing_rollouts,event_count,matched_clean_controls,control_rows,source_run_count",
                "safe,max_token_prob,1,1,0,[],1,1,1,1,1",
            ]) + chr(10),
        }
        for filename, content in files.items():
            (directory / filename).write_text(content, encoding="utf-8")

        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertTrue(analysis["available"])
        self.assertFalse(analysis["temporal_available"])
        self.assertTrue(analysis["event_triggered_available"])
        self.assertEqual(analysis["primary_analysis_type"], "event_triggered")
        self.assertEqual(
            analysis["primary_analysis_source"]["directory"],
            "outputs/baseline_signal_analysis/event_triggered_test",
        )
        event_triggered = analysis["event_triggered"]
        self.assertTrue(event_triggered["available"])
        self.assertEqual(event_triggered["source"]["selection_count"], 1)
        self.assertEqual(len(event_triggered["curves"]), 2)
        self.assertEqual(len(event_triggered["change_scores"]), 2)
        self.assertEqual(len(event_triggered["summary"]), 2)
        self.assertEqual(len(event_triggered["peak_events"]), 1)
        self.assertEqual(
            event_triggered["native_sampling"]["safe"]["max_token_prob"][
                "interval_median_frames"
            ],
            1.0,
        )

    def test_analysis_run_accepts_disjoint_rynnvalue_sources(self) -> None:
        self.install_fake_temporal_analyzer()
        roots = self.seed_analysis_runs()
        partial = self.root / "outputs" / "baselines" / "analysis_inputs" / "rynnvalue_partial"
        partial.mkdir(parents=True)
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "baseline": "rynnvalue",
            "selected_rollouts": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "created_at": "2026-08-28T00:00:00+00:00",
            "completed_at": "2026-08-28T00:00:01+00:00",
        }
        (partial / "run.json").write_text(json.dumps(metadata) + chr(10), encoding="utf-8")
        (partial / "jobs.jsonl").write_text("", encoding="utf-8")
        runs = dict(roots)
        runs["rynnvalue"] = [roots["rynnvalue"], str(partial.relative_to(self.root))]
        with self.request(
            "/api/analysis/run",
            {"scope": "primary_natural", "runs": runs, "output_label": "multi_rynn"},
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["run_source_counts"]["rynnvalue"], 2)
        self.assertTrue(job["allow_partial_coverage"])
        self.assertEqual(job["command"].count("--rynnvalue-run"), 2)
        final = self.wait_for_job("/api/analysis-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")

    def test_analysis_snapshot_exposes_localization_and_compacts_arrays(self) -> None:
        analysis_root = self.root / "outputs" / "baseline_signal_analysis"
        directory = analysis_root / "localized"
        directory.mkdir(parents=True)
        selection = self.root / "selection.json"
        selection.write_text(json.dumps({"selection": [{"id": self.rollout["id"]}]}), encoding="utf-8")
        manifest_hash = hashlib.sha256((self.root / "manifest.jsonl").read_bytes()).hexdigest()
        metadata = {
            "schema_version": 2,
            "selection": str(selection),
            "manifest_sha256": manifest_hash,
            "counts": {"rollouts": 1},
            "pre_window_frames": 2,
            "post_window_frames": 4,
            "background_stride_frames": 3,
            "native_sampling_preserved": True,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "event_metrics.jsonl").write_text("", encoding="utf-8")
        csv_files = {
            "method_coverage.csv": [
                "method,selected_rollouts,available_rollouts,missing_rollouts,run_status,run_completed_jobs,run_failed_jobs",
                "safe,1,1,0,complete,1,0",
            ],
            "summary_by_method_signal_outcome.csv": [
                "method,signal,outcome_group,n_events",
                "safe,max_token_prob,recovered_success,1",
            ],
            "summary_by_method_outcome.csv": [
                "method,outcome_group,n_events",
                "safe,recovered_success,1",
            ],
            "onset_signal_statistics.csv": [
                "method,signal,outcome_group,n_onset_events",
                "safe,max_token_prob,recovered_success,1",
            ],
            "clean_background_summary.csv": [
                "method,signal,n_pseudo_events",
                "safe,max_token_prob,10",
            ],
            "localization_summary.csv": [
                "method,signal,threshold,outcome_group,n_events,recall,false_alarm_rate,f1",
                "safe,max_token_prob,q95,all_events,1,1.0,0.1,0.5",
            ],
            "localization_thresholds.csv": [
                "method,signal,clean_score_q95",
                "safe,max_token_prob,2.0",
            ],
            "localization_by_failure_type.csv": [
                "method,signal,threshold,failure_type,recall",
                "safe,max_token_prob,q95,grasp_failure,1.0",
            ],
        }
        for filename, rows in csv_files.items():
            (directory / filename).write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
        localization_event = {
            "method": "safe",
            "signal": "max_token_prob",
            "threshold": "q95",
            "rollout_id": self.rollout["id"],
            "event_frame": 4,
            "hit": True,
            "sample_frames": [1, 2, 3],
        }
        (directory / "localization_event_metrics.jsonl").write_text(
            json.dumps(localization_event) + chr(10),
            encoding="utf-8",
        )
        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertTrue(analysis["localization_available"])
        self.assertEqual(len(analysis["localization_summary"]), 1)
        self.assertEqual(len(analysis["localization_thresholds"]), 1)
        self.assertEqual(len(analysis["localization_by_failure_type"]), 1)
        self.assertEqual(len(analysis["localization_event_metrics"]), 1)
        self.assertNotIn("sample_frames", analysis["localization_event_metrics"][0])
        self.assertEqual(analysis["localization_event_metrics"][0]["task_id"], 0)

    def test_analysis_dashboard_details_and_artifact_download(self) -> None:
        analysis_root = self.root / "outputs" / "baseline_signal_analysis"
        directory = analysis_root / "dashboard"
        directory.mkdir(parents=True)
        selection = self.root / "selection.json"
        selection.write_text(
            json.dumps({"selection": [{"id": self.rollout["id"]}]}),
            encoding="utf-8",
        )
        metadata = {
            "schema_version": 1,
            "analysis": "lf3r_baseline_change_points",
            "selection": str(selection),
            "manifest_sha256": hashlib.sha256(
                (self.root / "manifest.jsonl").read_bytes()
            ).hexdigest(),
            "generated_at": "2026-09-03T00:00:00+00:00",
            "counts": {"rollouts": 1, "observable_events": 1},
            "methods": ["safe"],
            "features": ["level"],
            "local_scales_frames": [16],
            "thresholds": ["q95"],
            "tolerances_frames": [4, 8, 16, 30],
        }
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        events = [
            json.dumps({
                "method": "safe",
                "signal": "max_token_prob",
                "feature": "level",
                "scale_frames": 16,
                "threshold": "q95",
                "outcome_group": "terminal_failure",
                "failure_type": "grasp_failure",
                "rollout_id": self.rollout["id"],
                "event_index": index,
                "event_frame": index + 2,
                "event_score": 0.2 + index,
                "hit": index == 0,
            })
            for index in range(3)
        ]
        (directory / "changepoint_event_metrics.jsonl").write_text(
            chr(10).join(events) + chr(10),
            encoding="utf-8",
        )
        files = {
            "method_coverage.csv": "method,selected_rollouts,available_rollouts,missing_rollouts" + chr(10) + "safe,1,1,0" + chr(10),
            "changepoint_summary.csv": (
                "method,signal,scale_frames,feature,threshold,outcome_group,n_events,recall" + chr(10)
                + "safe,max_token_prob,16,level,q95,all_events,1,1.0" + chr(10)
            ),
            "changepoint_reference_summary.csv": (
                "method,signal,scale_frames,feature,threshold_q95" + chr(10)
                + "safe,max_token_prob,16,level,1.0" + chr(10)
            ),
            "changepoint_by_failure_type.csv": (
                "method,signal,scale_frames,feature,threshold,failure_type,n_events,recall" + chr(10)
                + "safe,max_token_prob,16,level,q95,grasp_failure,1,1.0" + chr(10)
            ),
            "changepoint_scales.csv": (
                "method,signal,scale_frames,feature,threshold_q95" + chr(10)
                + "safe,max_token_prob,16,level,1.0" + chr(10)
            ),
        }

        for filename, content in files.items():
            (directory / filename).write_text(content, encoding="utf-8")

        with self.request("/api/analysis") as response:
            dashboard = json.load(response)["analysis"]
        self.assertTrue(dashboard["dashboard"])
        self.assertEqual(dashboard["default_scope"], "primary_natural")
        self.assertEqual(dashboard["detail_counts"]["changepoint_events"], 3)
        with self.request(
            "/api/analysis/details?kind=changepoint_events&page=1&page_size=2&sort=rollout"
        ) as response:
            details = json.load(response)
        self.assertTrue(details["available"])
        self.assertEqual(details["total"], 3)
        self.assertEqual(details["page_count"], 2)
        self.assertEqual(len(details["items"]), 2)
        with self.request("/api/analysis/artifacts/changepoint_summary.csv") as response:
            artifact = response.read().decode("utf-8")
        self.assertIn("max_token_prob", artifact)
        for query in (
            "/api/analysis/details?kind=unknown",
            "/api/analysis/details?page_size=101",
            "/api/analysis/artifacts/not_allowed.csv",
        ):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request(query)
            self.assertEqual(caught.exception.code, 400)

    def test_rollout_generation_validates_and_reports_job(self) -> None:
        self.install_fake_rollout_generator()
        invalid_payloads = [
            {"gpu": "0,1"},
            {"task_start": 4, "task_end": 3},
            {"task_start": 10},
            {"trials": 0},
            {"seed": -1},
            {"log_safe_features": "true"},
            {"run_label": "../bad"},
            {"unexpected": True},
        ]
        for payload in invalid_payloads:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/rollouts/generate", payload)
            self.assertEqual(caught.exception.code, 400)

        with self.request(
            "/api/rollouts/generate",
            {
                "gpu": "0",
                "task_start": 1,
                "task_end": 2,
                "trials": 2,
                "seed": 7,
                "run_label": "web_test",
            },
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["job_type"], "rollout_generation")
        self.assertTrue(job["persistent"])
        self.assertTrue(job["tmux_session"].startswith("lf3r-annotator-"))
        self.assertEqual(job["interpreter"], "bash")
        self.assertTrue(job["log_safe_features"])
        self.assertIn("--log-safe-features", job["command"])
        record_path = self.root / job["job_record_path"]
        self.assertTrue(record_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["tmux_session"], job["tmux_session"])
        wrapper = (self.root / record["wrapper_path"]).read_text(encoding="utf-8")
        self.assertIn("export MUJOCO_GL=egl", wrapper)
        self.assertIn("export PYOPENGL_PLATFORM=egl", wrapper)
        self.assertEqual(job["expected_rollouts"], 4)
        self.assertEqual(job["memory_gate"]["gpu_utilization"], "informational")
        self.assertEqual(job["memory_gate"]["minimum_free_mib"], 30720)
        self.assertTrue(job["run_note"].startswith("lf3r-data-natural-libero10-"))
        self.assertTrue(job["run_note"].endswith("-web_test"))
        final = self.wait_for_job("/api/rollout-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["completed_rollouts"], 4)
        self.assertTrue(final["manifest_rebuilt"])
        generator_args = (self.root / "fake_generator_args").read_text(encoding="utf-8").splitlines()
        self.assertEqual(generator_args[1:5], ["1", "2", "2", "7"])
        self.assertEqual(generator_args[6], "--log-safe-features")
        output_dir = self.root / final["output_dir"] / "libero_10"
        self.assertEqual(len(list(output_dir.glob("*.mp4"))), 4)
        with self.request("/api/rollout-jobs/" + job["job_id"] + "/log?tail=20") as response:
            log = json.load(response)["log"]
        self.assertIn("fake generator", log["text"])

        with self.request(
            "/api/rollouts/generate",
            {"task_start": 0, "task_end": 0, "run_label": "web_no_latent", "log_safe_features": False},
        ) as response:
            disabled_job = json.load(response)["job"]
        self.assertFalse(disabled_job["log_safe_features"])
        self.assertNotIn("--log-safe-features", disabled_job["command"])
        disabled_final = self.wait_for_job("/api/rollout-jobs", disabled_job["job_id"])
        self.assertEqual(disabled_final["status"], "complete")
        self.assertEqual((self.root / "fake_generator_args").read_text(encoding="utf-8").splitlines()[6], "")

    def test_rollout_generation_supports_spatial_native_suite(self) -> None:
        self.install_fake_spatial_rollout_generator()
        with self.request(
            "/api/rollouts/generate",
            {
                "task_suite": "libero_spatial",
                "gpu": "0",
                "task_start": 1,
                "task_end": 2,
                "trials": 2,
                "seed": 7,
                "run_label": "spatial_test",
            },
        ) as response:
            self.assertEqual(response.status, 202)
            job = json.load(response)["job"]
        self.assertEqual(job["task_suite"], "libero_spatial")
        self.assertEqual(job["suite_label"], "LIBERO-Spatial")
        self.assertEqual(job["expected_rollouts"], 4)
        self.assertEqual(job["render_resolution"], 256)
        self.assertEqual(job["policy_resolution"], 224)
        self.assertEqual(job["record_resolution"], 256)
        self.assertEqual(job["output_root"], "outputs/openvla_libero_spatial_native")
        self.assertTrue(job["run_note"].startswith("lf3r-data-natural-libero-spatial-256-"))
        self.assertTrue(job["log_safe_features"])
        self.assertIn("--log-safe-features", job["command"])
        self.assertIn("generate_libero_spatial_native.sh", job["command"][1])
        final = self.wait_for_job("/api/rollout-jobs", job["job_id"])
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["completed_rollouts"], 4)
        self.assertTrue(final["manifest_rebuilt"])
        self.assertEqual((self.root / "fake_spatial_generator_args").read_text(encoding="utf-8").splitlines()[6], "--log-safe-features")
        output_dir = self.root / final["output_dir"] / "libero_spatial"
        self.assertEqual(len(list(output_dir.glob("*.mp4"))), 4)

    def test_rollout_generation_memory_gate_and_shared_conflict(self) -> None:
        self.install_fake_rollout_generator(exit_code=75)
        with self.request("/api/rollouts/generate", {}) as response:
            job = json.load(response)["job"]
        final = self.wait_for_job("/api/rollout-jobs", job["job_id"])
        self.assertEqual(final["status"], "memory_blocked")
        self.assertEqual(final["return_code"], 75)
        self.assertFalse(final["manifest_rebuilt"])

        self.install_fake_rollout_generator()
        self.app.job_coordinator.acquire("another-compute-job")
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.request("/api/rollouts/generate", {"run_label": "conflict"})
            self.assertEqual(caught.exception.code, 409)
        finally:
            self.app.job_coordinator.release("another-compute-job")

    def test_analysis_environment_is_reported_and_missing_environment_blocks_run(self) -> None:
        with self.request("/api/health") as response:
            health = json.load(response)
        self.assertTrue(health["analysis_environment"]["ready"])
        self.assertTrue(
            health["analysis_environment"]["path"].endswith(
                "conda_envs/LF3R-ananlyse/bin/python"
            )
        )
        self.app.analysis_jobs.analysis_python = self.root / "missing-analysis-python"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/analysis/run", {})
        self.assertEqual(caught.exception.code, 503)

    def test_api_jobs_contains_persistent_job_records(self) -> None:
        runner = self.root / "tools" / "baselines" / "run_lf3r_baseline.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_text("# persistent fake runner\n", encoding="utf-8")
        with self.request(
            "/api/baselines/run-batch",
            {"baseline": "safe", "scope": "primary_natural", "gpu": "0"},
        ) as response:
            job = json.load(response)["job"]
        record_path = self.root / job["job_record_path"]
        self.assertTrue(record_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["tmux_session"], job["tmux_session"])
        self.assertEqual(record["interpreter"], job["interpreter"])
        with self.request("/api/jobs?job_type=baseline") as response:
            payload = json.load(response)
        self.assertTrue(any(item["job_id"] == job["job_id"] for item in payload["jobs"]))
        final = self.wait_for_job("/api/baseline-jobs", job["job_id"])
        self.assertIn(final["tmux_state"], {"exited", "running", "reattached"})

    def test_analysis_without_snapshot_is_available_but_marked_missing(self) -> None:
        with self.request("/api/analysis") as response:
            analysis = json.load(response)["analysis"]
        self.assertFalse(analysis["available"])
        self.assertIn("No complete baseline analysis snapshot", analysis["message"])

if __name__ == "__main__":
    unittest.main()
