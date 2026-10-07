"""Audit complete packs/splits through official loaders and one frozen loss forward."""
import argparse,json,gc
from pathlib import Path
import h5py,numpy as np,torch
from omegaconf import OmegaConf
from common import ROOT,write_json,sha256
from vera.datasets.registry import build_dataset
from vera.datasets.core.packed import load_packed_metadata,open_packed_npz,decode_packed_rgb_frame,decode_packed_flow_frame
from vera.experiments import build_experiment


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    manifest=json.loads((args.data/'split_manifest.json').read_text());targets=ROOT/manifest['targets']
    ids={s:{e['episode_id'] for e in manifest['episodes'] if e['split']==s} for s in ['training','validation','test']}
    assert not(ids['training']&ids['validation'] or ids['training']&ids['test'] or ids['validation']&ids['test'])
    cfg=OmegaConf.load(args.data/'configs/libero_idm.yaml');datasets={s:build_dataset(cfg[key],stage=s) for s,key in [('training','dataset'),('validation','validation_dataset'),('test','test_dataset')]}
    assert len(datasets['validation']) == len(ids['validation']) > 0
    assert len(datasets['test']) == len(ids['test']) > 0
    count=pairs=0;max_du_error=0.;quant_error=0.
    for ep in manifest['episodes']:
        path=args.data/ep['path'];r=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==r['packed_sha256']
        meta=load_packed_metadata(path);assert meta['num_frames']==ep['frames']
        dataset=datasets[ep['split']];position=next(i for i,e in enumerate(dataset._episodes) if Path(e.paths['packed_npz']).resolve()==path.resolve())
        episode=dataset._select_episode(position);indices=np.arange(ep['pairs']);du=dataset._derive_du(episode,indices).numpy()
        label=np.load(targets/ep['target_npz'])['action_idm'];np.testing.assert_allclose(du,label,atol=2e-6,rtol=2e-6)
        max_du_error=max(max_du_error,float(np.abs(du-label).max()));quant_error=max(quant_error,r['max_quantization_error'])
        npz=open_packed_npz(path)
        with h5py.File(ROOT/ep['source_hdf5'],'r') as source:
            group=source['data/'+ep['demo']]
            for view,original in zip(meta['views'],['agentview_rgb','eye_in_hand_rgb']):
                assert meta['flow_entries'][view]['num_frames']==ep['pairs']
                for i in [0,ep['pairs']-1]:
                    np.testing.assert_array_equal(decode_packed_rgb_frame(npz,i,view),group['obs/'+original][i])
                    flow=decode_packed_flow_frame(npz,i,view,'qint8_zstd_npz');assert flow.shape==(2,128,128) and np.isfinite(flow).all()
        count+=1;pairs+=ep['pairs']
        if count%50==0:print('PACKS_VERIFIED',count,flush=True)
    assert count==sum(manifest['counts'].values()) and pairs==sum(manifest['pairs'].values())
    experiment=build_experiment(cfg)
    module=experiment.data_module
    assert module._validation_dataset_cfg_nodes() is None
    assert module._select_dataset_cfg_node('test').data_root==cfg.test_dataset.data_root
    assert module._select_dataset_cfg_node('validation').data_root==cfg.validation_dataset.data_root
    sample=datasets['training'][0]
    assert sample['rgb'].shape==(8,2,3,128,128) and sample['du'].shape==(8,7) and sample['flow'].shape==(8,2,2,128,128)
    assert all(torch.isfinite(v).all() for v in sample.values())
    # Exact frozen algorithm used by the prepared config; no optimizer/backward/fit.
    algorithm=experiment._build_algo()
    checkpoint=ROOT/'checkpoints/vera-jidm/idm-mimicgen-285ouq1q/model.ckpt'
    ckpt=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    algorithm.load_state_dict(ckpt['state_dict'],strict=True);del ckpt;gc.collect()
    algorithm.eval().requires_grad_(False).cuda()
    batch={k:v.unsqueeze(0).cuda() for k,v in sample.items()}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        losses,metrics,total,output,diagnostics,timing=algorithm._compute_loss(batch,collect_diagnostics=False)
    assert torch.isfinite(total) and torch.isfinite(output.optical_flow).all()
    original=OmegaConf.load(targets/'original_checkpoint_config.yaml')
    assert list(cfg.dataset.action_abs_scale)==list(original.dataset.action_abs_scale)
    assert list(cfg.dataset.oflow_abs_scale)==list(original.dataset.oflow_abs_scale)
    result={'passed':True,'demos':count,'pairs':pairs,'split_counts':{k:len(v) for k,v in ids.items()},'max_du_error':max_du_error,
        'max_flow_quantization_error_pixels':quant_error,'split_disjoint':True,'lossless_rgb_parity':True,'test_route_verified':True,
        'sample_shapes':{k:list(v.shape) for k,v in sample.items()},'official_algorithm_strict_load':True,'frozen_forward_total_loss':float(total),
        'normalization_unchanged':True,'training_started':False,'backward_or_optimizer_steps':0,'wm_used':False}
    write_json(args.data/'verification.json',result)
    state=json.loads((args.data/'preparation.json').read_text());state['status']='READY';state['verification']=result;write_json(args.data/'preparation.json',state)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
