"""Pre-register deterministic trajectories from official LIBERO success HDF5."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from common import ROOT, CAMERAS, write_json, sha256, state_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--demos-per-task', type=int, default=5)
    args = ap.parse_args()
    assert args.output.resolve().is_relative_to(ROOT)
    tasks = []
    for path in sorted((ROOT / 'datasets/libero_official/libero_10').glob('*.hdf5')):
        with h5py.File(path, 'r') as f:
            data = f['data']
            demos = sorted(data, key=lambda n: int(n.split('_')[-1]))
            # Equal spacing, independent of actions/outcomes and model performance.
            selected = [demos[i] for i in np.linspace(0, len(demos) - 1, args.demos_per_task, dtype=int)]
            env_args = json.loads(data.attrs['env_args'])
            task = {'file': str(path.relative_to(ROOT)), 'task_name': path.stem.removesuffix('_demo'),
                    'bddl': str(data.attrs['bddl_file_name']), 'controller': env_args['env_kwargs']['controller_configs'],
                    'control_freq': env_args['env_kwargs']['control_freq'],
                    'image_convention': str(data.attrs.get('macros_image_convention', 'unknown')),
                    'demos': []}
            for demo in selected:
                g = data[demo]
                count = len(g['actions'])
                assert count > 1 and g['actions'].shape == (count, 7)
                assert bool(g['dones'][-1]) and bool(g['rewards'][-1])
                for cam in CAMERAS:
                    assert g['obs/' + cam].shape == (count, 128, 128, 3)
                assert len(g['states']) == count
                name = task['task_name'] + '__' + demo
                target = args.output / 'trajectories' / name
                target.mkdir(parents=True, exist_ok=True)
                # obs[k] is POST action[k], so the pair has GT action[k+1].
                pair_index = np.arange(count - 1, dtype=np.int64)
                np.savez_compressed(target / 'reference.npz', pair_index=pair_index,
                    action_index=pair_index + 1, state_index=pair_index + 1,
                    gt_action=g['actions'][1:], states=g['states'][:],
                    ee_pos=g['obs/ee_pos'][:], ee_ori=g['obs/ee_ori'][:],
                    gripper=g['obs/gripper_states'][:], joints=g['obs/joint_states'][:],
                    gt_metric=state_delta(g['obs/ee_pos'][:], g['obs/ee_ori'][:], g['obs/gripper_states'][:]))
                task['demos'].append({'demo': demo, 'id': name, 'frames': count, 'pairs': count - 1,
                                      'output': str(target.relative_to(args.output))})
            task['sha256'] = sha256(path)
            tasks.append(task)
    assert len(tasks) == 10
    checkpoint_dir = ROOT / 'checkpoints/vera-jidm/idm-mimicgen-285ouq1q'
    plan = {'schema': 1, 'suite': 'libero_10', 'demos_per_task': args.demos_per_task,
            'selection': 'np.linspace(0,n_demos-1,demos_per_task,dtype=int)',
            'checkpoint': str(checkpoint_dir.relative_to(ROOT)), 'tasks': tasks,
            'alignment': 'obs[k] is post-action[k]; pair(k,k+1) -> action[k+1], restore states[k+1]',
            'input': 'RGB uint8/255, native OpenGL orientation, external then wrist, no synthetic images',
            'wm_used': False, 'training': False, 'pair_batch': 8, 'tracker_grid': 15,
            'solver_lam': 0.01, 'normalized_grip_deadband': 0.18,
            'se3_scale': 50., 'gripper_scale': 80.}
    write_json(args.output / 'plan.json', plan)
    print('PLAN', len(tasks), 'tasks', sum(len(t['demos']) for t in tasks), 'demos',
          sum(d['pairs'] for t in tasks for d in t['demos']), 'pairs', flush=True)


if __name__ == '__main__':
    main()
