from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend_core import ValidationError
from repair.wan import WanAdapter
from repair.service import RepairService
from repair.batch import RepairRunGroupService


class WanRepairTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # External Wan paths live outside LF3R's project root.
        self.project = self.root / 'lf3r'
        self.project.mkdir()
        self.source = self.root / 'external-wan'
        self.source.mkdir()
        (self.source / 'generate.py').write_text('# official inference stub')
        self.checkpoint = self.root / 'weights'
        self.checkpoint.mkdir()
        (self.project / 'high.mp4').write_bytes(b'video')
        self.rollout = {'id': 'success', 'ground_truth_outcome': 'success',
                        'camera_video_paths': {'cam_high': 'high.mp4'},
                        'task_description': '  Put the cup on the table.\n',
                        'total_frames': 10, 'fps': 20}
        self.config = {'name': 'wan2_2', 'python': sys.executable,
                       'checkpoint': str(self.checkpoint), 'source_root': str(self.source)}
        self.service = RepairService(self.project, Mock(), Mock())
        self.payload = {'rollout_id': 'success', 'cut_type': 'frame', 'cut_frame': 4,
                        'gpu_index': 0, 'world_model': self.config}

    def test_wan_eligibility_without_actions_states_or_ctrl_preparation(self):
        row = self.service.catalog([self.rollout])[0]
        self.assertTrue(row['repair_eligible'])
        self.assertTrue(row['model_eligibility']['wan2_2']['eligible'])
        self.assertFalse(row['model_eligibility']['a2world']['eligible'])
        self.assertFalse(row['model_eligibility']['ctrl_world']['eligible'])
        self.assertFalse(row['actions_available'])
        self.assertFalse(row['sim_state_available'])
        with patch.object(self.service, '_worker_python', side_effect=AssertionError('LIBERO must not run')), patch('repair.service.run_alignment_subprocess', side_effect=AssertionError('no simulator')), patch.object(self.service, '_gpu_plan', return_value={'requested_index': 0}):
            plan = self.service.validate_with_alignment(self.payload, {'success': self.rollout})
        self.assertTrue(plan['ready'], plan['blockers'])
        self.assertEqual(plan['world_model']['instruction'], self.rollout['task_description'])
        self.assertEqual(plan['worker_python'], sys.executable)
        self.assertIsNone(plan['repair_runtime'])

    def test_instruction_fallback_and_invalid_inputs(self):
        for changes in ({'task_description': ''}, {'camera_video_paths': {}},
                        {'total_frames': 1}, {'ground_truth_outcome': 'failure'}):
            with self.subTest(changes=changes):
                rollout = {**self.rollout, **changes}
                with patch.object(self.service, '_gpu_plan', return_value={'requested_index': 0}):
                    try:
                        plan = self.service.validate_plan(self.payload, {'success': rollout})
                    except ValidationError:
                        continue
                self.assertFalse(plan['ready'])
        rollout = {**self.rollout, 'task_description': '', 'instruction': 'Original instruction'}
        self.assertEqual(WanAdapter(self.project, self.config).validate_rollout(rollout)['instruction'], 'Original instruction')
        with self.assertRaises(ValidationError):
            self.service.validate_plan({**self.payload, 'cut_frame': 9}, {'success': self.rollout})

    def test_runtime_validation(self):
        noexec = self.root / 'non-executable'
        noexec.write_text('python')
        for field, value, reason in [('python', '/missing/python', 'Wan Python'),
                                     ('python', str(noexec), 'not executable'),
                                     ('checkpoint', '/missing/checkpoint', 'checkpoint directory'),
                                     ('source_root', '/missing/source', 'generate.py')]:
            with self.subTest(field=field, value=value):
                status = WanAdapter(self.project, {**self.config, field: value}).validate_rollout(self.rollout)
                self.assertFalse(status['available'])
                self.assertIn(reason, '; '.join(status['unavailable_reasons']))
        (self.source / 'generate.py').unlink()
        self.assertFalse(WanAdapter(self.project, self.config).validate_rollout(self.rollout)['available'])

    def test_external_source_from_environment(self):
        config = {k: v for k, v in self.config.items() if k != 'source_root'}
        with patch.dict(os.environ, {'WAN_SOURCE_ROOT': str(self.source)}):
            self.assertTrue(WanAdapter(self.project, config).validate_rollout(self.rollout)['available'])
        local_source = self.project / 'repos/Wan2.2'
        local_source.mkdir(parents=True)
        (local_source / 'generate.py').write_text('# official source stub')
        with patch.dict(os.environ, {'WAN_SOURCE_ROOT': ''}):
            status = WanAdapter(self.project, config).validate_rollout(self.rollout)
        self.assertTrue(status['available'])
        self.assertEqual(status['source_root'], str(local_source))

    def test_official_command_and_failure_handling(self):
        adapter = WanAdapter(self.project, {**self.config, 'gpu_index': 3})
        adapter.validate_rollout(self.rollout)
        condition = {'image': str(self.project / 'condition.png'), 'instruction': self.rollout['task_description']}
        output_dir = self.project / 'generated'
        def run(command, **kwargs):
            self.assertEqual(command[command.index('--task') + 1], 'i2v-A14B')
            self.assertEqual(command[command.index('--prompt') + 1], condition['instruction'])
            for flag in ('--image', '--save_file', '--ckpt_dir'):
                self.assertTrue(Path(command[command.index(flag) + 1]).is_absolute())
            self.assertEqual(kwargs['cwd'], str(self.source))
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertNotIn('PYTHONPATH', kwargs['env'])
            self.assertFalse(any('action' in arg or 'state' in arg or 'prefix' in arg for arg in command))
            Path(command[command.index('--save_file') + 1]).write_bytes(b'video')
            return subprocess.CompletedProcess(command, 0)
        with patch('repair.wan.subprocess.run', side_effect=run):
            generated = adapter.generate(condition=condition, output_dir=output_dir)
        self.assertEqual(set(generated), {'cam_high'})
        generated['cam_high'].unlink()
        for result, message in [(subprocess.CompletedProcess([], 2), 'inference failed'),
                                (subprocess.CompletedProcess([], 0), 'output video is missing')]:
            with patch('repair.wan.subprocess.run', return_value=result), self.assertRaisesRegex(ValidationError, message):
                adapter.generate(condition=condition, output_dir=output_dir)

    def test_start_preserves_job_lifecycle_and_external_absolute_paths(self):
        self.service.tmux.submit_async.side_effect = lambda job, *args, **kwargs: job
        with patch.object(self.service, '_gpu_plan', return_value={'requested_index': 0}):
            job = self.service.start(self.payload, {'success': self.rollout})
        directory = self.project / job['run_dir']
        config = json.loads((directory / 'config.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        self.assertEqual(job['job_type'], 'repair_synthetic_suffix')
        self.assertEqual(config['world_model']['source_root'], str(self.source))
        self.assertTrue(config['generated_includes_condition'])
        self.assertFalse(provenance['runtime_libero_required'])
        self.assertEqual(provenance['generation_config']['task'], 'i2v-A14B')
        self.service.coordinator.acquire.assert_called_once()

    def test_export_exact_cut_rgb_instruction_and_metadata(self):
        import numpy as np
        import imageio.v2 as imageio
        from PIL import Image
        frame = np.full((16, 16, 3), 73, dtype=np.uint8)
        reader = Mock()
        reader.get_data.return_value = frame
        with patch.object(imageio, 'get_reader', return_value=reader):
            condition = WanAdapter(self.project, self.config).prepare_condition(self.rollout, cut_frame=4, output_dir=self.project / 'prepared')
        reader.get_data.assert_called_once_with(4)
        reader.close.assert_called_once()
        self.assertTrue(np.array_equal(np.asarray(Image.open(condition['image'])), frame))
        self.assertEqual(Path(condition['instruction_path']).read_text(), self.rollout['task_description'])
        self.assertTrue(Path(condition['image']).is_absolute())
        self.assertEqual(json.loads((self.project / 'prepared/wan_input.json').read_text()), condition)
        self.assertFalse(condition['future_actions_used'])
        self.assertFalse(condition['sim_state_used'])

    def test_group_preparation_keeps_wan_config_and_worker_environment(self):
        group = RepairRunGroupService(self.project, Mock(), Mock(), self.service)
        with patch.object(self.service, '_gpu_plan', return_value={'requested_index': 0}):
            plan = self.service.validate_plan(self.payload, {'success': self.rollout})
        item = group._prepare_run(payload=self.payload, rollout=self.rollout, plan=plan, run_group='repair-group-test')
        self.assertEqual(item['worker_python'], sys.executable)
        directory = self.project / item['run_dir']
        config = json.loads((directory / 'config.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        self.assertEqual(config['run_group'], 'repair-group-test')
        self.assertEqual(provenance['run_group'], 'repair-group-test')
        self.assertEqual(config['world_model']['source_root'], str(self.source))
        self.assertTrue(config['generated_includes_condition'])
        self.assertFalse(provenance['runtime_libero_required'])
        self.assertEqual(provenance['generation_config']['task'], 'i2v-A14B')

    def test_worker_complete_compare_history_and_failure_status(self):
        import imageio.v2 as imageio
        import numpy as np
        imageio.mimwrite(str(self.project / 'high.mp4'),
                         [np.full((16, 16, 3), i * 10, dtype=np.uint8) for i in range(10)], fps=20)
        self.service.tmux.submit_async.side_effect = lambda job, *args, **kwargs: job
        worker = Path(__file__).resolve().parents[1] / 'repair/worker.py'
        for fails in (False, True):
            with self.subTest(fails=fails), patch.object(self.service, '_gpu_plan', return_value={'requested_index': 0}):
                job = self.service.start(self.payload, {'success': self.rollout})
                directory = self.project / job['run_dir']
                def generate(adapter, *, condition, output_dir):
                    self.assertEqual(condition['instruction'], self.rollout['task_description'])
                    if fails:
                        raise ValidationError('Wan2.2 inference failed with exit code 2')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output = output_dir / 'cam_high.mp4'
                    imageio.mimwrite(str(output), [np.full((16, 16, 3), 40, dtype=np.uint8)] * 5, fps=16)
                    return {'cam_high': output}
                with patch.object(WanAdapter, 'generate', generate), patch('non_analysis_tools.gpu_status', return_value={'available': False, 'gpus': []}), patch('repair.trajectory.load_actions', side_effect=AssertionError('actions must not load')), patch('repair.alignment_runner.run_alignment_subprocess', side_effect=AssertionError('LIBERO must not run')), patch.object(sys, 'argv', [str(worker), '--project-root', str(self.project), '--run-dir', str(directory)]):
                    if fails:
                        with self.assertRaisesRegex(ValidationError, 'inference failed'):
                            runpy.run_path(str(worker), run_name='__main__')
                    else:
                        runpy.run_path(str(worker), run_name='__main__')
                status = json.loads((directory / 'status.json').read_text())
                self.assertEqual(status['status'], 'failed' if fails else 'complete')
                if fails:
                    self.assertIn('inference failed', status['error'])
                    continue
                provenance = json.loads((directory / 'provenance.json').read_text())
                self.assertIsNone(provenance['gt_action_end'])
                self.assertIsNone(provenance['gt_future_action_count'])
                self.assertTrue(provenance['standardized_generated_includes_condition'])
                detail = self.service.detail(job['run_id'])
                self.assertEqual(set(detail['videos']['generated']), {'cam_high'})
                self.assertEqual(set(detail['videos']['real']), {'cam_high'})
                self.assertEqual(detail['videos']['generated_fps'], 16)
                self.assertEqual(detail['videos']['generated_source_offset_seconds'], 0)
                self.assertEqual(detail['metrics']['views']['cam_high']['generated_frames'], 5)
                self.assertEqual(self.service.list_runs()[0]['world_model'], 'wan2_2')

    def test_existing_model_adapter_selection(self):
        for name, expected in [('a2world', 'a2world'), ('Ctrl-World', 'ctrl_world'), ('Wan2.2-I2V-A14B', 'wan2_2')]:
            self.assertEqual(self.service._world_model_adapter(name, {})[0], expected)


class WanFrontendTest(unittest.TestCase):
    def test_model_option_and_payload_in_javascript(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('node unavailable')
        static = Path(__file__).resolve().parents[1] / 'static/repair'
        self.assertIn('<option value="wan2_2">Wan2.2-I2V-A14B</option>', (static / 'page.js').read_text())
        source = (static / 'synthetic-suffix.js').read_text()
        source = source.replace('  installEvents();', '  window.testApi = {buildPayload: buildPayload, modelEligibility: modelEligibility};')
        harness = '''
const vm = require('vm');
const assert = require('assert');
const values = {repairWorldModel:'wan2_2',repairCutType:'frame',repairCutFrame:'4',repairGpu:'2',repairAlignmentPsnr:'20',repairWanPython:'/external/bin/python',repairWanCheckpoint:'/external/weights',repairWanSourceRoot:'/external/wan'};
const context = {window:{addEventListener(){}},document:{getElementById(id){return {value:values[id]};}}};
vm.runInNewContext(SOURCE, context);
const row = {id:'success',frames:10,repair_eligible:true,model_eligibility:{wan2_2:{eligible:true},a2world:{eligible:false}}};
context.window.LF3RRepairSyntheticSuffix.state.rollouts=[row];
context.window.LF3RRepairSyntheticSuffix.state.selectedRolloutId='success';
const payload=context.window.testApi.buildPayload();
assert.equal(payload.world_model.name,'wan2_2');
assert.equal(payload.world_model.python,'/external/bin/python');
assert.equal(payload.world_model.checkpoint,'/external/weights');
assert.equal(payload.world_model.source_root,'/external/wan');
assert.equal(payload.cut_frame,4);
assert.equal(payload.gpu_index,2);
assert.equal(Object.keys(payload.world_model).length,4);
assert.equal(context.window.testApi.modelEligibility(row).eligible,true);
values.repairWorldModel='a2world';
assert.equal(context.window.testApi.modelEligibility(row).eligible,false);
'''.replace('SOURCE', json.dumps(source))
        batch_source = (static / 'batch-ui.js').read_text().replace('  var installed = false;', '  window.testBatch = {commonPayload: commonPayload};\n  var installed = false;')
        harness += "\ncontext.window.setTimeout=function(){};\nvm.runInNewContext(" + json.dumps(batch_source) + ", context);\nvalues.repairWorldModel='wan2_2';\nconst batch=context.window.testBatch.commonPayload();\nassert.equal(batch.world_model.name,'wan2_2');\nassert.equal(batch.world_model.python,'/external/bin/python');\nassert.equal(batch.generated_includes_condition,true);\n"
        result = subprocess.run([node, '-e', harness], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
