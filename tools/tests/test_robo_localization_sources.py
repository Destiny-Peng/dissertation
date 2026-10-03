"""Source selection checks, with no inference or training."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robo_localization_head import specs, spec_runner


class LocalizationSourceTests(unittest.TestCase):
    def test_old_and_separate_source_specs(self):
        old = specs.normalize_spec({'name': 'old', 'base': {}})
        self.assertEqual(old['base']['data']['success_source_run_root'], '')
        new = specs.normalize_spec({'name': 'new', 'base': {'data': {
            'source_run_root': 'outputs/failure',
            'success_source_run_root': 'outputs/success'}}})
        self.assertEqual(new['base']['data']['success_source_run_root'], 'outputs/success')
        specs.validate_config(new['base'])
        new['base']['data']['success_source_run_root'] = 42
        with self.assertRaises(ValueError):
            specs.validate_config(new['base'])

    def test_cohort_filters_before_loading_other_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fused = {key: {'outcome': outcome, 'source_run_root': root,
                          'frames': [4], 'progress': [0.1], 'hops': [0.1]}
                     for key, outcome in [('f', 'terminal_failure'), ('s', 'clean_success'),
                                          ('r', 'recovered_success')]}
            events = [{'rollout_id': 'f'}, {'rollout_id': 'r'}]
            clean = [{'rollout_id': 's'}]
            for cohort, selected, expected_events, expected_clean in [
                ('failure', {'f'}, events[:1], []), ('success', {'s'}, [], clean)]:
                def unavailable(*args, **kwargs):
                    raise FileNotFoundError('test modality missing')
                with mock.patch.object(spec_runner, 'build_base_records', return_value=(
                    fused, events, [], clean, {'usable_rollout_n': 3})), \
                     mock.patch.object(spec_runner, 'load_signal', side_effect=unavailable) as load:
                    records, provenance = spec_runner._aligned_signal_records(root, {}, root, cohort=cohort)
                self.assertEqual(set(records['fused'][0]), selected)
                self.assertEqual(records['fused'][1], expected_events)
                self.assertEqual(records['fused'][3], expected_clean)
                self.assertEqual(provenance['fused']['usable_rollout_n'], 1)
                self.assertTrue(all(call.args[1] in selected for call in load.call_args_list))

    def test_merges_separate_runs_with_source_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure, success = root/'failure', root/'success'
            event, clean = {'rollout_id': 'f'}, {'rollout_id': 's'}
            def aligned(source, *args, **kwargs):
                is_failure = source == failure
                self.assertEqual(kwargs['cohort'], 'failure' if is_failure else 'success')
                key = 'f' if is_failure else 's'
                records = ({key: {'source_run_root': source}}, [event] if is_failure else [],
                           [], [] if is_failure else [clean])
                modes = {name: records for name in ('fused', 'robodopamine_latent')}
                return modes, {'fused': {'completed_rollout_n': 1},
                               'modes': {name: {} for name in modes}}
            audit = root/'audit.json'
            with mock.patch.object(spec_runner, '_aligned_signal_records', side_effect=aligned), \
                 mock.patch.object(spec_runner, 'project_relative', side_effect=str):
                modes, provenance = spec_runner._analysis_source_records(failure, success, {}, root, audit)
            self.assertEqual(set(modes['fused'][0]), {'f', 's'})
            self.assertEqual(modes['fused'][1], [event])
            self.assertEqual(modes['fused'][3], [clean])
            self.assertEqual(modes['robodopamine_latent'][0]['s']['source_run_root'], success)
            self.assertEqual(provenance['selection_mode'], 'separate_failure_success_runs')
            self.assertEqual(json.loads(audit.read_text())['sources']['success']['root'], str(success))
            self.assertEqual(provenance['fused']['usable_rollout_n'], 2)

    def test_empty_success_ratio_errors_without_training(self):
        records = {'fused': ({'f': {}}, [{'rollout_id': 'f'}], [], [])}
        config = {'data': {'population': 'failure_success', 'success_ratio': 0.2}}
        with self.assertRaisesRegex(ValueError, 'Positive success_ratio'):
            spec_runner._records_for_config(config, records)
        config['data']['success_ratio'] = 0
        self.assertEqual(spec_runner._records_for_config(config, records), records['fused'])
        config['data']['success_ratio'] = 0.2
        config['data']['population'] = 'failure_only'
        spec_runner._records_for_config(config, records)
        config['data']['population'] = 'failure_success'
        records['fused'] = (*records['fused'][:3], [{'rollout_id': 's'}])
        spec_runner._records_for_config(config, records)

    def test_shared_run_keeps_legacy_selection(self):
        root = Path('same_run')
        with mock.patch.object(spec_runner, '_aligned_signal_records', return_value=({}, {})) as load:
            for success in (None, root):
                spec_runner._analysis_source_records(root, success, {}, root, root/'audit.json')
            self.assertEqual(load.call_count, 2)
            self.assertTrue(all(not call.kwargs for call in load.call_args_list))


if __name__ == '__main__':
    unittest.main()
