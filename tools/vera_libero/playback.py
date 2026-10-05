"""LIBERO simulator validation and replay in LF3R's existing simulation environment."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT, read_plan, write_json
sys.path.insert(0, str(ROOT / 'repos/LIBERO'))


def relocate_xml(xml):
    """Keep recorded dynamics/cameras exactly; relocate only mesh/texture files."""
    import robosuite
    root = ET.fromstring(xml)
    for node in root.iter():
        old = node.get('file')
        if not old:
            continue
        parts = Path(old).parts
        if 'robosuite' in parts:
            at = max(i for i, part in enumerate(parts) if part == 'robosuite')
            new = Path(robosuite.__file__).parent.joinpath(*parts[at + 1:])
        elif 'assets' in parts:
            at = parts.index('assets')
            new = ROOT / 'repos/LIBERO/libero/libero/assets'
            new = new.joinpath(*parts[at + 1:])
        else:
            raise ValueError('Unknown XML asset path: ' + old)
        assert new.is_file(), str(new)
        assert new.resolve().is_relative_to(ROOT)
        node.set('file', str(new))
    return ET.tostring(root, encoding='unicode')


def build_env(task, xml, camera=False):
    from libero.libero.envs import OffScreenRenderEnv
    bddl = ROOT / 'repos/LIBERO' / task['bddl']
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=['agentview', 'robot0_eye_in_hand'],
        camera_heights=128, camera_widths=128, control_freq=task['control_freq'],
        use_camera_obs=camera, ignore_done=True, horizon=10000, render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")))
    env.seed(0)
    env.reset()
    env.reset_from_xml_string(relocate_xml(xml))
    env.sim.forward()
    ctrl = env.robots[0].controller
    np.testing.assert_allclose(ctrl.output_max, task['controller']['output_max'])
    np.testing.assert_allclose(ctrl.output_min, task['controller']['output_min'])
    return env


def restore(env, state):
    obs = env.set_init_state(state)
    robot = env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    # MuJoCo's flattened state omits the PandaGripper Python target accumulator.
    # Initialize that accumulator from observed finger qpos, not a GT command.
    q = np.asarray(obs['robot0_gripper_qpos'])
    joint_ids = [env.sim.model.joint_name2id(name) for name in robot.gripper.joints]
    rng = env.sim.model.jnt_range[joint_ids]
    normalized = 2 * (q - rng[:, 0]) / (rng[:, 1] - rng[:, 0]) - 1
    robot.gripper.current_action = np.clip(normalized, -1., 1.)
    return obs


def pose(obs):
    return np.asarray(obs['robot0_eef_pos']), Rotation.from_quat(obs['robot0_eef_quat']), np.asarray(obs['robot0_gripper_qpos'])


def errors(obs, ref, frame):
    p, r, g = pose(obs)
    return float(np.linalg.norm(p - ref['ee_pos'][frame])), float((r * Rotation.from_rotvec(ref['ee_ori'][frame]).inv()).magnitude()), float(np.linalg.norm(g - ref['gripper'][frame]))


def check_alignment(env, g, ref, target):
    count = len(ref['ee_pos']) - 1
    samples = sorted(set([0, count // 2, count - 1]))
    rows = []
    for k in samples:
        obs = restore(env, ref['states'][k + 1])
        row = {'pair_index': k, 'restored_state_index': k + 1,
               'current_pose_error': list(errors(obs, ref, k)), 'cameras': {}}
        for saved, live in [('agentview_rgb', 'agentview'), ('eye_in_hand_rgb', 'robot0_eye_in_hand')]:
            rendered = env.sim.render(height=128, width=128, camera_name=live)
            reference = g['obs/' + saved][k]
            mse = float(np.square(rendered.astype(np.float64) - reference).mean())
            flipped_mse = float(np.square(rendered[::-1].astype(np.float64) - reference).mean())
            psnr = 100. if mse == 0 else float(10 * np.log10(255. ** 2 / mse))
            row['cameras'][saved] = {'psnr_raw': psnr, 'mse_raw': mse, 'mse_flip_y': flipped_mse}
        nxt, _, _, _ = env.step(ref['gt_action'][k])
        row['next_pose_error'] = list(errors(nxt, ref, k + 1))
        rows.append(row)
    # Recorded post-action RGB/proprio and corresponding state must agree.
    passed = all(r['current_pose_error'][0] < 0.003 and r['current_pose_error'][1] < 0.01
                 and all(c['psnr_raw'] >= 20. for c in r['cameras'].values()) for r in rows)
    report = {'passed': passed, 'samples': rows,
              'alignment': 'obs[k] -> obs[k+1] uses actions[k+1], restored from states[k+1]',
              'xml_restored': True, 'asset_paths_only_modified': True,
              'tolerances': {'position_m': 0.003, 'rotation_rad': 0.01, 'image_psnr_db': 20.}}
    write_json(target / 'alignment.json', report)
    assert passed, 'Stored observation/state alignment failed; refuse inference/playback interpretation'
    return report


def free_playback(env, actions, ref, target, label, make_video):
    obs = restore(env, ref['states'][1])
    n = len(actions)
    states, poses, all_errors, success_flags = [], [], [], []
    writer = imageio.get_writer(str(target / (label + '.mp4')), fps=4, codec='libx264') if make_video else None
    try:
        for k, action in enumerate(actions):
            obs, _, _, _ = env.step(action)
            states.append(env.get_sim_state())
            p, r, grip = pose(obs)
            poses.append(np.concatenate([p, r.as_rotvec(), grip]))
            all_errors.append(errors(obs, ref, k + 1))
            success_flags.append(bool(env.check_success()))
            if writer and (k % 5 == 0 or k == n - 1):
                frame = np.concatenate([env.sim.render(height=128, width=128, camera_name=c)[::-1]
                                        for c in ['agentview', 'robot0_eye_in_hand']], axis=1)
                writer.append_data(frame)
        np.savez_compressed(target / (label + '_playback.npz'), states=np.asarray(states),
            poses=np.asarray(poses), pose_errors=np.asarray(all_errors), success=np.asarray(success_flags))
        err = np.asarray(all_errors)
        return {'steps': n, 'final_success': success_flags[-1], 'ever_success': any(success_flags),
                'mean_position_error_m': float(err[:, 0].mean()), 'final_position_error_m': float(err[-1, 0]),
                'mean_rotation_error_rad': float(err[:, 1].mean()), 'mean_gripper_error_m': float(err[:, 2].mean())}
    finally:
        if writer:
            writer.close()


def one_step_playback(env, pred, ref, target):
    rows = []
    gt_rows = []
    for k, action in enumerate(pred):
        restore(env, ref['states'][k + 1])
        obs, _, _, _ = env.step(action)
        rows.append(errors(obs, ref, k + 1))
        restore(env, ref['states'][k + 1])
        obs, _, _, _ = env.step(ref['gt_action'][k])
        gt_rows.append(errors(obs, ref, k + 1))
    e, g = np.asarray(rows), np.asarray(gt_rows)
    np.savez_compressed(target / 'one_step_playback.npz', predicted_errors=e, gt_errors=g,
                        pair_index=ref['pair_index'])
    return {'steps': len(pred), 'pred_mean_position_error_m': float(e[:, 0].mean()),
            'gt_mean_position_error_m': float(g[:, 0].mean()),
            'pred_mean_rotation_error_rad': float(e[:, 1].mean()),
            'gt_mean_rotation_error_rad': float(g[:, 1].mean()),
            'pred_mean_gripper_error_m': float(e[:, 2].mean()),
            'gt_mean_gripper_error_m': float(g[:, 2].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--alignment-only', action='store_true')
    args = ap.parse_args()
    plan = read_plan(args.output)
    for task in plan['tasks']:
        with h5py.File(ROOT / task['file'], 'r') as f:
            for i, demo in enumerate(task['demos']):
                target = args.output / demo['output']
                alignment_ok = (target / 'alignment.json').exists() and json.loads((target / 'alignment.json').read_text())['passed']
                if args.alignment_only and alignment_ok:
                    continue
                if not args.alignment_only and (target / 'playback_metrics.json').exists():
                    continue
                started = time.time()
                g = f['data/' + demo['demo']]
                ref = np.load(target / 'reference.npz')
                xml = g.attrs['model_file']
                if isinstance(xml, bytes):
                    xml = xml.decode()
                env = build_env(task, xml)
                try:
                    if not alignment_ok:
                        if (target / 'alignment.json').exists():
                            (target / 'alignment.json').rename(target / ('alignment_failed_' + str(time.time_ns()) + '.json'))
                        check_alignment(env, g, ref, target)
                    if args.alignment_only:
                        print('ALIGNMENT_PASS', demo['id'], flush=True)
                        continue
                    pred = np.load(target / 'prediction.npz')['predicted_action']
                    results = {'gt': free_playback(env, ref['gt_action'], ref, target, 'gt', i == 0),
                               'idm': free_playback(env, pred, ref, target, 'idm', i == 0),
                               'one_step': one_step_playback(env, pred, ref, target),
                               'seconds': time.time() - started,
                               'protocol': 'offline teacher-forced RGB pair prediction; free playback is open loop, no future simulator GT injection'}
                    actual_gt = np.load(target / 'gt_playback.npz')
                    actual_idm = np.load(target / 'idm_playback.npz')
                    pos_err = np.linalg.norm(actual_idm['poses'][:, :3] - actual_gt['poses'][:, :3], axis=1)
                    rot_err = (Rotation.from_rotvec(actual_idm['poses'][:, 3:6]) *
                               Rotation.from_rotvec(actual_gt['poses'][:, 3:6]).inv()).magnitude()
                    grip_err = np.linalg.norm(actual_idm['poses'][:, 6:] - actual_gt['poses'][:, 6:], axis=1)
                    np.savez_compressed(target / 'paired_actual_motion.npz', position_error_m=pos_err,
                                        rotation_error_rad=rot_err, gripper_error_m=grip_err,
                                        state_error=actual_idm['states'] - actual_gt['states'])
                    results['paired_actual_motion'] = {'mean_position_error_m': float(pos_err.mean()),
                        'final_position_error_m': float(pos_err[-1]), 'mean_rotation_error_rad': float(rot_err.mean()),
                        'mean_gripper_error_m': float(grip_err.mean())}
                    write_json(target / 'playback_metrics.json', results)
                    print('PLAYBACK_DONE', demo['id'], json.dumps(results), flush=True)
                finally:
                    env.close()
    print('ALIGNMENT_COMPLETE' if args.alignment_only else 'PLAYBACK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
