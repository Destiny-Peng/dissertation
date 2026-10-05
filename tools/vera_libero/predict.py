"""Official frozen MimicGen VGGT J-IDM + official CoTracker3, true two-frame inputs."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from common import ROOT, CAMERAS, DIMENSIONS, read_plan, write_json, sha256, metric_to_osc, metrics


def checkpoint_model(plan, device):
    import vera.idm
    from vera.idm.jacobian.models.registry import resolve_model_cfg, resolve_model_instance
    path = ROOT / plan['checkpoint']
    release = json.loads((path / 'release_files.json').read_text())
    entry = next(x for x in release if x['path'].endswith('/model.ckpt'))
    expected = entry.get('lfs', {}).get('oid')
    actual = sha256(path / 'model.ckpt')
    assert actual == expected, (actual, expected)
    cfg = OmegaConf.load(path / 'config.yaml')
    node = OmegaConf.to_container(cfg.algorithm.model, resolve=True)
    # The official task checkpoint bundles every VGGT backbone tensor; avoid a
    # redundant download or overwrite of the trained backbone.
    node['pretrained_model_id'] = None
    node['checkpoint_path'] = None
    ckpt = torch.load(path / 'model.ckpt', map_location='cpu', mmap=True, weights_only=False)
    sd = ckpt['state_dict']
    model_state = {}
    extras = []
    for k, v in sd.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
            if k.startswith('_orig_mod.'):
                k = k[len('_orig_mod.'):]
            model_state[k] = v
        else:
            extras.append(k)
    # The release initialized VGGT with from_pretrained (native 518px learned
    # pos_embed), despite the IDM's 128px input configuration. Preserve that
    # architecture while using the bundled weights; runtime RGB stays 128px.
    position_count = model_state['vggt.aggregator.patch_embed.pos_embed'].shape[1] - 1
    patch_side = int(round(position_count ** 0.5))
    assert patch_side * patch_side == position_count
    node['image_size'] = patch_side * int(node['patch_size'])
    model = resolve_model_instance(resolve_model_cfg(node))
    # Reject silent random initialization or dropped learned tensors.
    assert not extras, extras[:10]
    model.load_state_dict(model_state, strict=True)
    params = sum(p.numel() for p in model.parameters())
    del ckpt, model_state, sd
    gc.collect()
    model.eval().requires_grad_(False).to(device)
    return model, cfg, {'sha256': actual, 'parameters': params, 'strict_load': True,
                       'model_cfg': node, 'dataset_norm': OmegaConf.to_container(cfg.dataset, resolve=True)}


def tracker_model(device):
    sys.path.insert(0, str(ROOT / 'repos/co-tracker'))
    from cotracker.predictor import CoTrackerPredictor
    path = ROOT / 'checkpoints/vera-jidm/cotracker3/scaled_offline.pth'
    tracker = CoTrackerPredictor(checkpoint=None, offline=True, v2=False, window_len=60)
    sd = torch.load(path, map_location='cpu', weights_only=False)
    tracker.model.load_state_dict(sd, strict=True)
    tracker.eval().requires_grad_(False).to(device)
    return tracker, {'sha256': sha256(path), 'name': 'cotracker3_offline', 'grid_size': 15,
                     'source': 'facebook/cotracker3/scaled_offline.pth', 'strict_load': True}


@torch.inference_mode()
def pair_predict(model, tracker, current, following, flow_scale, action_scale, lam, grid):
    from vera.idm.jacobian.models.base import InputObservation
    from vera.policy.motion_policy_types import tikhonov_solve
    batch, views, h, w, _ = current.shape
    rgb = torch.from_numpy(current.copy()).to(next(model.parameters()).device)
    rgb = rgb.permute(0, 1, 4, 2, 3).float() / 255.
    nxt = torch.from_numpy(following.copy()).to(rgb.device).permute(0, 1, 4, 2, 3).float() / 255.
    # Matches the released server's independent-view compute_jacobian path.
    j = model.compute_jacobian(InputObservation(rgb=rgb.reshape(batch * views, 3, h, w)))
    j = j.float().reshape(batch, views, 7, 2, h, w)
    assert torch.isfinite(j).all(), 'nonfinite model output'
    video = torch.stack((rgb, nxt), dim=2).reshape(batch * views, 2, 3, h, w) * 255.
    xy, visibility = tracker(video, grid_size=grid)
    xy = xy.float().reshape(batch, views, 2, -1, 2)
    vis = visibility.reshape(batch, views, 2, -1)
    src = xy[:, :, 0]
    disp = xy[:, :, 1] - src
    valid = vis[:, :, 0] & vis[:, :, 1]
    valid &= torch.isfinite(src).all(-1) & torch.isfinite(disp).all(-1)
    valid &= (xy[..., 0] >= 0).all(2) & (xy[..., 0] <= w - 1).all(2)
    valid &= (xy[..., 1] >= 0).all(2) & (xy[..., 1] <= h - 1).all(2)
    valid &= (disp[..., 0].abs() < w - 1) & (disp.norm(dim=-1) < w - 1)
    safe = torch.nan_to_num(src).round().long()
    idx = safe[..., 1].clamp(0, h - 1) * w + safe[..., 0].clamp(0, w - 1)
    # b,v,pixel,xy,command: same nearest-pixel gather as official VERA.
    field = j.permute(0, 1, 4, 5, 3, 2).reshape(batch, views, h * w, 2, 7)
    sparse = field.gather(2, idx[..., None, None].expand(-1, -1, -1, 2, 7))
    target = disp / flow_scale.view(1, 1, 1, 2)
    weights = valid.float().unsqueeze(-1).expand(-1, -1, -1, 2)
    a = (sparse * weights[..., None]).reshape(batch, -1, 7)
    b = (target * weights).reshape(batch, -1)
    # Float64 tiny solve for diagnostics; official solver also moves to CPU.
    u = tikhonov_solve(a.cpu().double(), b.cpu().double(), lam=lam)
    assert torch.isfinite(u).all(), 'nonfinite solve'
    residual = (a.cpu().double() @ u.unsqueeze(-1)).squeeze(-1) - b.cpu().double()
    condition = torch.linalg.cond(a.cpu().double().transpose(-1, -2) @ a.cpu().double())
    return u.numpy(), {'valid_tracks': valid.sum((1, 2)).cpu().numpy(),
                      'flow_rmse': residual.square().mean(1).sqrt().numpy(),
                      'condition': np.nan_to_num(condition.numpy(), posinf=1e30),
                      'mean_disp': disp.norm(dim=-1).mean((1, 2)).cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')  # matches official ImageJacobian._build_model
    torch.manual_seed(0)
    plan = read_plan(args.output)
    for task in plan['tasks']:
        for demo in task['demos']:
            alignment = args.output / demo['output'] / 'alignment.json'
            assert alignment.exists() and json.loads(alignment.read_text())['passed'], str(alignment)
    model, cfg, model_meta = checkpoint_model(plan, args.device)
    tracker, tracker_meta = tracker_model(args.device)
    assert not any(name.startswith('vera.video_model') for name in sys.modules)
    write_json(args.output / 'model_load.json', {'idm': model_meta, 'tracker': tracker_meta,
               'torch': torch.__version__, 'device': args.device, 'precision': 'float32 weights; official high matmul precision (TF32 allowed)',
               'jacobian_multiview': 'independent views, matches official serving',
               'predictor_future_frames': 1, 'wm_used': False})
    flow_scale = torch.tensor(cfg.dataset.oflow_abs_scale, device=args.device)
    action_scale = np.asarray(cfg.dataset.action_abs_scale)
    for task in plan['tasks']:
        with h5py.File(ROOT / task['file'], 'r') as f:
            for demo in task['demos']:
                target = args.output / demo['output']
                if (target / 'prediction.npz').exists():
                    print('RESUME', demo['id'], flush=True)
                    continue
                started = time.time()
                g = f['data/' + demo['demo']]
                ref = np.load(target / 'reference.npz')
                n = demo['pairs']
                demo_rgb = np.stack([g['obs/' + c][:] for c in CAMERAS], axis=1)
                predictions, diagnostics = [], []
                for start in range(0, n, args.batch):
                    end = min(start + args.batch, n)
                    curr = demo_rgb[start:end]
                    next_obs = demo_rgb[start + 1:end + 1]
                    u, diag = pair_predict(model, tracker, curr, next_obs, flow_scale, action_scale,
                                           plan['solver_lam'], plan['tracker_grid'])
                    predictions.append(u)
                    diagnostics.append(diag)
                    if start % 40 == 0:
                        print('PAIR', demo['id'], start, '/', n, 'seconds', round(time.time() - started, 1), flush=True)
                norm = np.concatenate(predictions)
                scaled = norm * (action_scale + 1e-8) / float(cfg.dataset.du_scale)
                metric = scaled / np.asarray([50.] * 6 + [80.])
                raw, pred = metric_to_osc(metric, ref['ee_pos'][:-1], ref['gripper'][:-1], norm[:, 6], task['controller'])
                gt = ref['gt_action']
                diag = {k: np.concatenate([d[k] for d in diagnostics]) for k in diagnostics[0]}
                tmp = target / 'prediction.tmp.npz'
                np.savez_compressed(tmp, predicted_action=pred, predicted_action_unclipped=raw,
                    gt_action=gt, error=pred - gt, absolute_error=np.abs(pred - gt),
                    squared_error=(pred - gt) ** 2, normalized_prediction=norm,
                    predicted_metric_delta=metric, gt_metric_delta=ref['gt_metric'],
                    metric_error=metric - ref['gt_metric'], pair_index=ref['pair_index'],
                    action_index=ref['action_index'], **diag)
                tmp.replace(target / 'prediction.npz')
                result = {'action': metrics(pred, gt), 'unclipped_action': metrics(raw, gt),
                          'state_delta': metrics(metric, ref['gt_metric']),
                          'zero_action': metrics(np.zeros_like(gt), gt),
                          'zero_state_delta': metrics(np.zeros_like(metric), ref['gt_metric']),
                          'arm_action': metrics(pred[:, :6], gt[:, :6]),
                          'gripper_accuracy': float((pred[:, 6] == gt[:, 6]).mean()),
                          'clipped_fraction': float((np.abs(raw[:, :6]) > 1).mean()),
                          'seconds': time.time() - started}
                write_json(target / 'prediction_metrics.json', result)
                with (target / 'actions.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['pair_index', 'action_index'] + ['pred_' + d for d in DIMENSIONS] +
                                    ['gt_' + d for d in DIMENSIONS] + ['error_' + d for d in DIMENSIONS])
                    for i in range(n):
                        writer.writerow([i, i + 1, *pred[i], *gt[i], *(pred[i] - gt[i])])
                print('DONE', demo['id'], json.dumps(result['action']), flush=True)
    print('PREDICTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
