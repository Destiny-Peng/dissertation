"""CPU-only latent extraction/experiment tests; no model inference or training."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/baselines"))
sys.path.insert(0, str(ROOT / "tools/lf3r_annotator"))
from robo_localization_head import latent, specs, spec_runner, core
from backend_core import ValidationError
from baseline_jobs import BaselineJobsMixin
from analysis_localization_results import AnalysisLocalizationResultsMixin
from baselines import posthoc_robo_localization
import robo_dopamine_persistent_worker as worker

module_spec = importlib.util.spec_from_file_location("robo_latent_capture", ROOT / "repos/Robo-Dopamine/examples/latent.py")
capture = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(capture)


class LatentTests(unittest.TestCase):
    def test_final_opening_tag_token_and_capture_alignment(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **_: ''.join(['<', 'score', '>', '+', '50', '%', '</score>', 'EOS'][i] for i in ids))
        ids = list(range(8))
        self.assertEqual(capture.score_token_index(tokenizer, ids), 2)
        # Opening tag can be a single token or split across tokens.
        single = SimpleNamespace(decode=lambda ids, **_: ''.join(['<score>', '+', '50'][i] for i in ids))
        self.assertEqual(capture.score_token_index(single, [0, 1, 2]), 0)
        self.assertEqual(capture.score_token_index(tokenizer, ids, "score_end"), 6)
        with self.assertRaises(ValueError):
            capture.select_features([SimpleNamespace(request_id='missing', outputs=[SimpleNamespace(token_ids=ids)])], [{}], tokenizer, 'score_start')
        with self.assertRaises(ValueError):
            capture.score_token_index(tokenizer, [0, 1])
        model = SimpleNamespace(compute_logits=lambda hidden: hidden + 1)
        runner = SimpleNamespace(model=model,
            input_batch=SimpleNamespace(req_ids=['rB', 'rA'], num_prompt_tokens=np.array([7, 9])),
            seq_lens=SimpleNamespace(np=np.array([10, 12])),
            vllm_config=SimpleNamespace(speculative_config=None, parallel_config=SimpleNamespace(pipeline_parallel_size=1)))
        engine = SimpleNamespace(model_runner=runner)
        capture.install_capture(engine)
        capture.start_capture(engine)
        hidden = torch.stack([torch.full((2560,), 2.), torch.full((2560,), 3.)])
        torch.testing.assert_close(model.compute_logits(hidden), hidden + 1)
        saved = capture.finish_capture(engine)
        # seq_len-1-prompt_len = 2: hidden for '>', used to predict '+' next.
        np.testing.assert_array_equal(saved['rB'][2], hidden[0].numpy())
        outputs = [SimpleNamespace(request_id='rA', outputs=[SimpleNamespace(token_ids=ids)])]
        values, indices = capture.select_features(outputs, [saved], tokenizer, 'score_start')
        self.assertEqual(indices, [2])
        np.testing.assert_array_equal(values[0], hidden[1].numpy())
        capture.start_capture(engine)
        self.assertEqual(capture.finish_capture(engine), {})

    def test_named_worker_rpc_uses_real_safe_vllm_serialization(self):
        import os
        from vllm.v1.serial_utils import MsgpackEncoder, MsgpackDecoder
        from vllm.v1.engine import UtilityOutput, UtilityResult
        from vllm.utils import resolve_obj_by_qualname
        with mock.patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            encoder = MsgpackEncoder()
            decoder = MsgpackDecoder()
            # Reproduce the original initialization failure without loading a model.
            with self.assertRaisesRegex(TypeError, 'not serializable'):
                encoder.encode(capture.install_capture)
            sys.path.insert(0, str(ROOT / 'repos/Robo-Dopamine'))
            try:
                extension = resolve_obj_by_qualname('examples.latent.LatentWorkerExtension')
            finally:
                sys.path.remove(str(ROOT / 'repos/Robo-Dopamine'))
            class FakeWorker(extension):
                pass
            engine = FakeWorker()
            runner = SimpleNamespace(model=SimpleNamespace(compute_logits=lambda x: x),
                input_batch=SimpleNamespace(req_ids=['r'], num_prompt_tokens=np.array([5])),
                seq_lens=SimpleNamespace(np=np.array([8])),
                vllm_config=SimpleNamespace(speculative_config=None,
                    parallel_config=SimpleNamespace(pipeline_parallel_size=1)))
            engine.model_runner = runner
            for method in ['lf3r_install_latent_capture', 'lf3r_start_latent_capture']:
                wire = encoder.encode((method, None, (), {}))
                method, _timeout, args, kwargs = decoder.decode(wire)
                getattr(engine, method)(*args, **kwargs)
            hidden = torch.arange(2560, dtype=torch.float32).reshape(1, 2560)
            runner.model.compute_logits(hidden)
            result = engine.lf3r_finish_latent_capture()
            # Exercise the actual typed engine utility response protocol too.
            response = UtilityOutput(call_id=1, result=UtilityResult([result]))
            decoded = MsgpackDecoder(UtilityOutput).decode(encoder.encode(response))
            np.testing.assert_array_equal(decoded.result.result[0]['r'][2], hidden[0].numpy())

    def test_saved_sample_and_frame_alignment(self):
        rows = [{'id': 'step-run-0000-bf_000000-af_000004'}, {'id': 'step-run-0001-bf_000004-af_000008'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latent_features.npz'
            capture.save_features(path, rows, [np.ones(2560), np.ones(2560)*2], [3, 4], 'score_start')
            x = latent.load_latent(path, [4, 8], [row['id'] for row in rows])
            old = Path(directory) / 'old_latent_features.npz'
            capture.save_features(old, rows, [np.ones(2560), np.ones(2560)], [3, 3], 'score_next')
            with self.assertRaisesRegex(ValueError, 're-extract'):
                latent.load_latent(old, [4, 8])
            self.assertEqual(x.shape, (2, 2560))
            with self.assertRaises(ValueError):
                latent.load_latent(path, [8, 4])
            with self.assertRaises(ValueError):
                latent.load_latent(path, [4, 8], ['wrong', 'ids'])

    def test_train_only_pca_and_fused_dimensions(self):
        rng = np.random.default_rng(3)
        dataset = {key: {'sequence': rng.normal(size=(6, 2562)).astype(np.float32)}
                   for key in ['train', 'val', 'test']}
        original = copy.deepcopy(dataset)
        pca = latent.fit_pca(dataset, ['train'], components=3)
        dataset['val']['sequence'][:] = 1e6
        dataset['test']['sequence'][:] = -1e6
        changed = latent.fit_pca(dataset, ['train'], components=3)
        np.testing.assert_array_equal(pca['components'], changed['components'])
        np.testing.assert_array_equal(pca['mean'], original['train']['sequence'][:, :2560].mean(axis=0))
        transformed = latent.transform_dataset(original, pca, True)
        self.assertEqual(transformed['test']['sequence'].shape, (6, 5))
        np.testing.assert_array_equal(transformed['test']['sequence'][:, -2:], original['test']['sequence'][:, -2:])
        self.assertEqual(latent.transform_dataset(original, pca)['test']['sequence'].shape, (6, 3))
        with self.assertRaises(ValueError):
            latent.fit_pca(dataset, ['train'], 128)

    def test_presets_and_launch_options(self):
        presets = AnalysisLocalizationResultsMixin._localization_builtin_presets()
        self.assertEqual(specs.normalize_spec({})['base']['data']['pca_components'], 64)
        self.assertEqual(presets['fused_progress_bilstm']['base']['data']['pca_components'], 64)
        for dimensions in (32, 64, 128):
            for suffix, mode in [('latent', 'robodopamine_latent'), ('latent_plus_fused', 'robodopamine_latent_plus_fused')]:
                name = f'pca{dimensions}_{suffix}_bilstm'
                normalized = specs.normalize_spec(presets[name])
                self.assertEqual(normalized['base']['data']['signal_mode'], mode)
                self.assertEqual(normalized['base']['data']['pca_components'], dimensions)
                self.assertEqual(normalized['base']['training']['split_seed'], 17)
        service = BaselineJobsMixin()
        self.assertTrue(service._validate_options('robo_dopamine', {'robo_extract_latent': True})['robo_extract_latent'])
        with self.assertRaises(ValidationError):
            service._validate_options('robo_dopamine', {'robo_extract_latent': True, 'robo_eval_mode': 'forward'})

    def test_aligned_training_records_load_same_run_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / 'pred_vllm.json'
            rows = [{'id': 'step-run-0000-bf_000000-af_000004'}, {'id': 'step-run-0001-bf_000004-af_000008'}]
            prediction.write_text(json.dumps(rows))
            capture.save_features(root/'latent_features.npz', rows, [np.ones(2560), np.ones(2560)*2], [3, 3], 'score_start')
            fused = {'frames': [4, 8], 'progress': [0.1, 0.2], 'hops': [0.1, 0.1], 'source_run_root': root}
            inc = {**fused, 'prediction_path': prediction}
            with mock.patch.object(spec_runner, 'build_base_records', return_value=({'r': fused}, [], [], [], {})), \
                 mock.patch.object(spec_runner, 'load_signal', return_value=inc):
                (root / "run.json").write_text("{}")
                records, provenance = spec_runner._aligned_signal_records(root, {}, root/'annotations')
                self.assertFalse(spec_runner.build_base_records.call_args.kwargs["latest_per_rollout"])
            self.assertEqual(records['robodopamine_latent'][0]['r']['features'].shape, (2, 2560))
            self.assertEqual(records['robodopamine_latent_plus_fused'][0]['r']['features'].shape, (2, 2562))
            np.testing.assert_array_equal(records['robodopamine_latent_plus_fused'][0]['r']['features'][:, -2:], core.sequence_from_signal(fused))
            self.assertEqual(provenance['modes']['robodopamine_latent']['excluded_rollout_n'], 0)

    def test_missing_annotations_are_reported_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = {"completed_rollout_n": 15, "excluded_rollouts": [
                {"rollout_id": f"r{i}", "reason": "annotation_missing"} for i in range(15)]}
            with mock.patch.object(spec_runner, "build_base_records", return_value=({}, [], [], [], provenance)):
                with self.assertRaisesRegex(ValueError, "annotation_missing.*15"):
                    spec_runner._aligned_signal_records(root, {}, root, root/"audit.json")
            saved = json.loads((root/"audit.json").read_text())
            self.assertEqual(saved["fused"]["completed_rollout_n"], 15)
        with self.assertRaisesRegex(ValueError, "source_run_root"):
            spec_runner._records_for_config({"data": {"signal_mode": "robodopamine_latent"}}, {"fused": ({"r": {}}, [], [], [])})

    def test_checkpoint_consumers_load_pca_without_training(self):
        for mode, dim in [('robodopamine_latent', 128), ('robodopamine_latent_plus_fused', 130)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'head.pt'
                model = core.TinyBiLSTM(hidden=2, input_dim=dim)
                pca = {'mean': np.zeros(2560, dtype=np.float32), 'components': np.eye(128, 2560, dtype=np.float32)}
                torch.save({'config': {'model': {'hidden': 2}, 'data': {'signal_mode': mode}},
                            'input_dim': dim, 'latent_pca': pca, 'model_state_dict': model.state_dict(),
                            'normalization_mean': torch.zeros(dim), 'normalization_std': torch.ones(dim)}, path)
                for loader in [worker._load_localization_checkpoint, posthoc_robo_localization.checkpoint_bundle]:
                    bundle = loader(path)
                    self.assertEqual(bundle['signal_mode'], mode)
                    self.assertEqual(bundle['latent_pca']['components'].shape, (128, 2560))

    def test_pipeline_preserves_generation_and_extracts_incremental_only(self):
        import ast
        import datetime
        import os
        import re
        import shutil
        import types
        tree = ast.parse((ROOT/'repos/Robo-Dopamine/examples/inference.py').read_text())
        chosen = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                  and node.name in {'GRMInference', 'build_samples_json'}]
        ns = {'List': list, 'Dict': dict, 'Path': Path, 'json': json, 're': re,
              'os': os, 'shutil': shutil, 'datetime': datetime.datetime,
              'SYSTEM_PROMPT': next(ast.literal_eval(node.value) for node in tree.body
                                    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'SYSTEM_PROMPT'),
              'ensure_dir': lambda p: p.mkdir(parents=True, exist_ok=True),
              'get_frame_count': lambda p: ('video', 9),
              'make_sample_indices_by_interval': lambda n, interval: [0, 4, 8],
              'save_frames': lambda *args: None, 'tqdm': lambda value: value,
              'Image': SimpleNamespace(open=lambda p: SimpleNamespace(convert=lambda mode: str(p)))}
        exec(compile(ast.Module(body=chosen, type_ignores=[]), 'inference.py', 'exec'), ns)
        init_calls = []
        def llm_factory(**kwargs):
            init_calls.append(kwargs)
            return SimpleNamespace(collective_rpc=lambda method: init_calls.append(method))
        ns.update(LLM=llm_factory, SamplingParams=lambda **kwargs: kwargs,
                  AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace()))
        ns['GRMInference']('mock-model', extract_latent=True)
        self.assertEqual(init_calls[0]['worker_extension_cls'], 'examples.latent.LatentWorkerExtension')
        self.assertEqual(init_calls[1], 'lf3r_install_latent_capture')
        init_calls.clear()
        ns['GRMInference']('mock-model', extract_latent=False)
        self.assertNotIn('worker_extension_cls', init_calls[0])
        self.assertEqual(len(init_calls), 1)
        instance = ns['GRMInference'].__new__(ns['GRMInference'])
        instance.extract_latent = True
        instance.latent_position = 'score_start'
        instance._capture_batch = False
        instance.sampling_params = object()
        instance.processor = SimpleNamespace(
            apply_chat_template=lambda messages, **kw: json.dumps(messages),
            tokenizer=SimpleNamespace(decode=lambda ids, **kw: ''.join(['<score>', '+', '25', '%', '</score>', 'EOS'][i] for i in ids)))
        calls = []
        saved = {}
        def rpc(method):
            self.assertIsInstance(method, str)
            calls.append(method)
            return [saved.copy()] if method == "lf3r_finish_latent_capture" else [None]
        def generate(prompts, **kwargs):
            self.assertIs(kwargs['sampling_params'], instance.sampling_params)
            self.assertFalse(kwargs['use_tqdm'])
            outputs = []
            for i, prompt in enumerate(prompts):
                self.assertEqual(len(prompt['multi_modal_data']['image']), 8)
                request_id = str(i)
                saved[request_id] = {0: [float(i)]*2560}
                outputs.append(SimpleNamespace(request_id=request_id,
                    outputs=[SimpleNamespace(text='<score>+25%</score>', token_ids=list(range(6)))]))
            return outputs
        instance.model = SimpleNamespace(collective_rpc=rpc, generate=generate)
        package = types.ModuleType('examples')
        package.__path__ = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {'examples': package, 'examples.latent': capture}):
            root = Path(directory)
            goal = root/'goal.png'
            goal.write_bytes(b'placeholder')
            args = dict(cam_high_path='a', cam_left_path='b', cam_right_path='c', out_root=str(root),
                        task='test task', goal_image=str(goal), batch_size=2)
            inc = Path(instance.run_pipeline(**args, eval_mode='incremental'))
            latent.load_latent(inc/'latent_features.npz', [4, 8])
            rows = json.loads((inc/'pred_vllm.json').read_text())
            self.assertEqual([row['progress'] for row in rows], [0.25, 0.4375])
            self.assertEqual(calls, ['lf3r_start_latent_capture', 'lf3r_finish_latent_capture'])
            calls.clear()
            forward = Path(instance.run_pipeline(**args, eval_mode='forward'))
            self.assertFalse((forward/'latent_features.npz').exists())
            self.assertEqual(calls, [])

    def test_training_integration_fits_pca_per_split_and_saves_checkpoint(self):
        rng = np.random.default_rng(2)
        signals = {f'r{i}': {'frames': list(range(150)),
                   'features': rng.normal(size=(150, 2562)).astype(np.float32),
                   'task_key': 'task', 'task_id': 0} for i in range(4)}
        events = [{'rollout_id': key, 'outcome': 'terminal_failure', 'causal_onset_frame': 2,
                   'observable_onset_frame': 3, 'task_key': 'task', 'task_id': 0} for key in signals]
        for dimensions, mode, dim in [(d, mode, d + extra) for d in (32, 64, 128)
                                     for mode, extra in [('robodopamine_latent', 0), ('robodopamine_latent_plus_fused', 2)]]:
            config = specs.deep_merge(specs.DEFAULT_BASE, {'data': {'signal_mode': mode, 'pca_components': dimensions}, 'training': {'device': 'cpu'}})
            train_meta = dict(best_epoch=0, best_val_loss=0.1, effective_train_batch_size=2, optimizer_steps_per_epoch=1)
            def fake_train(**kwargs):
                self.assertEqual(kwargs['input_dim'], dim)
                self.assertEqual(next(iter(kwargs['dataset'].values()))['sequence'].shape[1], dim)
                return core.TinyBiLSTM(hidden=16, input_dim=dim), train_meta
            def fake_logits(model, dataset, ids, *args):
                return {key: torch.zeros(len(dataset[key]['frames'])) for key in ids}
            with tempfile.TemporaryDirectory() as directory, \
                 mock.patch.object(core, 'train_bilstm', side_effect=fake_train) as train, \
                 mock.patch.object(core, 'batched_logits', side_effect=fake_logits):
                path = Path(directory)/'checkpoints'/'main'
                _, _, _, records = spec_runner._run_configuration(config=config, config_id='c1', stage_name='main',
                    repeats=2, signals=signals, events=events, no_event_failures=[], clean_rollouts=[], checkpoint_root=path)
                self.assertEqual(train.call_count, 2)
                for repeat, record in enumerate(records):
                    saved = torch.load(path/'c1'/f'repeat_{repeat:02d}.pt', weights_only=False)
                    self.assertEqual(saved['input_dim'], dim)
                    self.assertEqual(saved['latent_pca']['components'].shape, (dimensions, 2560))
                    self.assertEqual(saved['latent_pca']['fit_rollout_ids'], record['failure_train_ids'])
                    train_frames = np.concatenate([signals[i]['features'][:, :2560] for i in record['failure_train_ids']])
                    np.testing.assert_allclose(saved['latent_pca']['mean'], train_frames.mean(axis=0), atol=1e-6)
                    self.assertEqual(saved['latent_pca']['fit_sample_count'], len(train_frames))


if __name__ == '__main__':
    unittest.main()
