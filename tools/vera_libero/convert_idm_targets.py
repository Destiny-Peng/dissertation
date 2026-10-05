"""Export LIBERO achieved-motion supervision with the unchanged official IDM contract."""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import h5py
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from common import ROOT, CAMERAS, state_delta, write_json, sha256


def official_config(checkpoint):
    from vera.datasets.base import DatasetConfig
    dataset = yaml.safe_load((checkpoint / 'config.yaml').read_text())['dataset']
    # Missing fields have precisely the upstream DatasetConfig defaults.
    cfg = DatasetConfig(layout='separate', n_frames=dataset['sampling']['num_frames'],
        height=dataset['resolution'], per_view_w=dataset['resolution'],
        derive_action=True, linearize=dataset['sampling']['linearize_time_length'],
        se3_scale=dataset.get('se3_scale', 50.), gripper_scale=dataset.get('gripper_scale', 80.),
        du_scale=dataset['du_scale'], action_normalization_mode=dataset['action_normalization_mode'],
        action_abs_scale=dataset['action_abs_scale'], action_dim=7,
        action_pos_key='ee_pos', action_quat_key='ee_quat', action_gripper_key='gripper_states')
    assert cfg.linearize == 1 and cfg.action_normalization_mode == 'symmetric_percentile'
    return cfg, dataset


class TrajectoryLoader:
    def __init__(self, arrays): self.arrays = arrays
    def load_trajectory(self, episode, key): return self.arrays[key]


def convert(pos, ori, gripper, cfg):
    from vera.datasets.base import UnifiedDataset
    from vera.datasets.core.actions import SE3QuatDeltaAction
    quat = Rotation.from_rotvec(ori).as_quat()  # xyzw, same as upstream scipy convention
    loader = TrajectoryLoader({'ee_pos':pos, 'ee_quat':quat, 'gripper_states':gripper})
    holder = SimpleNamespace(cfg=cfg, action_model=SE3QuatDeltaAction(), view_loader=loader)
    # Invoke original derivation AND original float32 normalization, not an OSC approximation.
    du = UnifiedDataset._derive_du(holder, None, np.arange(len(pos)-1)).numpy()
    metric = state_delta(pos, ori, gripper)
    scales = np.asarray([cfg.se3_scale]*6+[cfg.gripper_scale], dtype=np.float32)
    scaled = torch.as_tensor(metric*scales, dtype=torch.float32) * cfg.du_scale
    independent = (scaled / (torch.tensor(cfg.action_abs_scale, dtype=torch.float32)+1e-8)).numpy()
    np.testing.assert_allclose(du, independent, atol=2e-6, rtol=2e-6)
    recovered = du.astype(np.float64)*(np.asarray(cfg.action_abs_scale)+1e-8)/cfg.du_scale/scales
    np.testing.assert_allclose(recovered, metric, atol=2e-8, rtol=2e-6)
    assert np.isfinite(du).all()
    return quat, metric, scaled.numpy(), du, float(np.max(np.abs(du-independent)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'datasets/libero_official/libero_10')
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/vera-jidm/idm-mimicgen-285ouq1q')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    assert args.output.resolve().is_relative_to(ROOT/'datasets')
    assert not (args.output/'manifest.json').exists(), 'Choose a new output to preserve prior conversion'
    cfg,dataset=official_config(args.checkpoint);episodes=[];all_du=[];max_diff=0.
    for path in sorted(args.source.glob('*.hdf5')):
        file_hash=sha256(path)
        with h5py.File(path,'r') as f:
            names=sorted(f['data'],key=lambda n:int(n.split('_')[-1]))
            for name in names:
                g=f['data/'+name];n=len(g['actions'])
                assert bool(g['rewards'][-1]) and bool(g['dones'][-1])
                assert g['actions'].shape==(n,7)
                for camera in CAMERAS: assert g['obs/'+camera].shape==(n,128,128,3)
                pos=g['obs/ee_pos'][:];ori=g['obs/ee_ori'][:];grip=g['obs/gripper_states'][:]
                quat,metric,scaled,du,diff=convert(pos,ori,grip,cfg);max_diff=max(max_diff,diff)
                rel=Path(path.stem)/ (name+'.npz');target=args.output/rel;target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,action_idm=du,action_idm_scaled=scaled,achieved_metric=metric,
                    action_libero_osc=g['actions'][1:],obs_index=np.arange(n-1),next_obs_index=np.arange(1,n),
                    action_index=np.arange(1,n),restore_state_index=np.arange(1,n),
                    ee_pos=pos,ee_quat=quat,gripper_states=grip)
                episodes.append({'source_hdf5':str(path.relative_to(ROOT)), 'source_hdf5_sha256':file_hash,
                    'demo':name,'frames':n,'pairs':n-1,'target_npz':str(rel),'target_sha256':sha256(target)})
                all_du.append(du)
        print('CONVERTED',path.stem,len(names),'demos',flush=True)
    du=np.concatenate(all_du)
    assert len(episodes)==500
    assert not any(x.startswith('vera.video_model') for x in sys.modules)
    manifest={'completed_at':datetime.now().astimezone().isoformat(),'suite':'libero_10','demos':len(episodes),'pairs':len(du),
        'source':str(args.source.relative_to(ROOT)),'checkpoint_config':str((args.checkpoint/'config.yaml').relative_to(ROOT)),
        'checkpoint_config_sha256':sha256(args.checkpoint/'config.yaml'),'official_target_config':asdict(cfg),
        'model_io_unchanged':True,'command_dim':7,'target_semantics':['translation(T1 @ inv(T0)) x/y/z','rotvec(R1 @ inv(R0)) x/y/z','left_finger_qpos[t+1]-left_finger_qpos[t]'],
        'action_transform':'official SE3QuatDeltaAction + UnifiedDataset._derive_du; SE3*50, finger*80, du_scale, then checkpoint symmetric normalization; no clipping',
        'alignment':'pair(obs[k],obs[k+1]) -> achieved motion obs[k] to obs[k+1], associated OSC action[k+1] and state[k+1]; action[0] excluded',
        'input':{'camera_order':list(CAMERAS),'resolution':[128,128],'rgb_scale':'uint8 / 255','image_orientation':'native OpenGL','official_window_layout':'[T,V,3,H,W]'},
        'flow_normalization':{k:dataset[k] for k in ['oflow_abs_scale','flow_normalization_mode','flow_normalization_space','flow_scale_factor']},
        'validation':{'passed':True,'official_vs_independent_max_error':max_diff,'physical_roundtrip_passed':True,'all_finite':True},
        'normalized_target_statistics':{'mean':du.mean(0).tolist(),'std':du.std(0).tolist(),'abs_max':np.abs(du).max(0).tolist(),'fraction_abs_above_one':(np.abs(du)>1).mean(0).tolist()},
        'training_started':False,'wm_used':False,'flow_generated':False,'episodes':episodes}
    write_json(args.output/'manifest.json',manifest)
    (args.output/'original_checkpoint_config.yaml').write_bytes((args.checkpoint/'config.yaml').read_bytes())
    readme='''# LIBERO targets in the original VERA MimicGen IDM format

Labels use the original 7D achieved-motion contract and original checkpoint normalization. Source HDF5 images are referenced, not copied or changed. Each NPZ contains normalized action_idm [T-1,7], scaled target, physical SE3/finger delta, original associated OSC actions and exact observation/action/state indices. Rotations use xyzw quaternions; SE3 is world-left T1 inv(T0), not plain Cartesian position subtraction. Gripper is achieved left-finger displacement, not binary command.

No label clipping: symmetric-percentile normalization does not require [-1,1]. The original checkpoint config is copied byte-for-byte. Dataset-specific percentile statistics are not refitted. All labels are validated against an independent SE3 calculation and a physical-unit round trip.

For training, use RGB in [T,V,3,H,W], original cameras/resolution and the saved du. Official J-IDM flow supervision still needs motion tracks/flow and the original flow normalization. This export contains targets/provenance only; it is not yet a complete packed training dataset, nor a newly trained model. Trajectory splits have not been selected. Keep entire demonstrations together when selecting them.

OSC commands alone cannot be converted exactly to achieved motion by scaling: controller state, lag, contact and gripper target history affect actual motion. This export uses official demonstrations' observed achieved motion to construct the exact original target. Conversely, predicted achieved motion is not guaranteed to reproduce the original OSC command; the simulator action adapter/controller must still be evaluated.
'''
    (args.output/'README.md').write_text(readme)
    print('TARGET_CONVERSION_COMPLETE',len(episodes),'demos',len(du),'pairs','max_reference_error',max_diff,flush=True)


if __name__=='__main__':main()
