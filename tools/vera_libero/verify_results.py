"""Audit completed artifacts against original official actions and simulator evidence."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
from common import ROOT, read_plan, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read_plan(args.output)
    demos = pairs = videos = 0
    max_restore_error, min_psnr = 0., float('inf')
    for task in plan['tasks']:
        with h5py.File(ROOT / task['file'], 'r') as source:
            for i, demo in enumerate(task['demos']):
                folder = args.output / demo['output']
                n = demo['pairs']
                pred = np.load(folder / 'prediction.npz')
                np.testing.assert_array_equal(pred['gt_action'], source['data/' + demo['demo'] + '/actions'][1:])
                np.testing.assert_array_equal(pred['action_index'], np.arange(1, n + 1))
                np.testing.assert_array_equal(pred['pair_index'], np.arange(n))
                assert pred['predicted_action'].shape == (n, 7)
                assert np.isfinite(pred['predicted_action']).all()
                assert np.isfinite(pred['predicted_metric_delta']).all()
                assert (np.abs(pred['predicted_action']) <= 1).all()
                np.testing.assert_allclose(pred['error'], pred['predicted_action'] - pred['gt_action'])
                np.testing.assert_allclose(pred['absolute_error'], np.abs(pred['error']))
                np.testing.assert_allclose(pred['squared_error'], pred['error'] ** 2)
                with (folder / 'actions.csv').open() as stream:
                    assert sum(1 for _ in csv.reader(stream)) == n + 1
                playback = json.loads((folder / 'playback_metrics.json').read_text())
                for label in ['gt', 'idm']:
                    actual = np.load(folder / (label + '_playback.npz'))
                    assert len(actual['states']) == len(actual['poses']) == len(actual['success']) == n
                    assert np.isfinite(actual['states']).all() and np.isfinite(actual['poses']).all()
                    assert bool(actual['success'][-1]) == playback[label]['final_success']
                single = np.load(folder / 'one_step_playback.npz')
                assert single['predicted_errors'].shape == single['gt_errors'].shape == (n, 3)
                assert np.isfinite(single['predicted_errors']).all() and np.isfinite(single['gt_errors']).all()
                assert len(np.load(folder / 'paired_actual_motion.npz')['position_error_m']) == n
                alignment = json.loads((folder / 'alignment.json').read_text())
                assert alignment['passed']
                for row in alignment['samples']:
                    max_restore_error = max(max_restore_error, row['current_pose_error'][0])
                    min_psnr = min(min_psnr, *(c['psnr_raw'] for c in row['cameras'].values()))
                if i == 0:
                    for label in ['gt', 'idm']:
                        with imageio.get_reader(folder / (label + '.mp4')) as reader:
                            frame = reader.get_data(0)
                            assert frame.shape == (128, 256, 3)
                        videos += 1
                demos += 1
                pairs += n
    summary = json.loads((args.output / 'summary.json').read_text())
    assert summary['demos'] == demos == 50 and summary['pairs'] == pairs == 13885
    loaded = json.loads((args.output / 'model_load.json').read_text())
    assert loaded['idm']['strict_load'] and loaded['tracker']['strict_load'] and not loaded['wm_used']
    result = {'passed': True, 'demos': demos, 'pairs': pairs, 'videos_decoded': videos,
              'gt_action_source_and_index_match': True, 'finite_and_error_arrays_verified': True,
              'full_playback_lengths_and_success_flags_verified': True,
              'max_restored_position_error_m': max_restore_error, 'min_camera_psnr_db': min_psnr}
    write_json(args.output / 'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
