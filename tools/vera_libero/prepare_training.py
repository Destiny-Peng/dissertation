"""Freeze trajectory splits and unchanged-I/O official J-IDM training configuration."""
import argparse,copy,json
from pathlib import Path
import numpy as np,yaml
from common import ROOT,write_json,sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert args.output.resolve().is_relative_to(ROOT/'datasets')
    m=json.loads((args.targets/'manifest.json').read_text());rng=np.random.default_rng(42);groups={}
    for e in m['episodes']:groups.setdefault(e['source_hdf5'],[]).append(e)
    eps=[];test_ids={0,12,24,36,49}
    for task,es in sorted(groups.items()):
        val_ids=set(rng.choice([i for i in range(50) if i not in test_ids],5,replace=False).tolist())
        for e in es:
            idx=int(e['demo'].split('_')[-1]);split='test' if idx in test_ids else 'validation' if idx in val_ids else 'training'
            episode_id=Path(task).stem+'__'+e['demo'];rel=str(Path('episodes')/(episode_id+'.npz'))
            eps.append({**e,'episode_id':episode_id,'split':split,'path':rel})
    for split in ['training','validation','test']:
        dest=args.output/split;dest.mkdir(parents=True,exist_ok=True)
        # Relative paths point to one canonical pack; no duplicate large artifacts.
        write_json(dest/'index.json',[str(Path('..')/e['path']) for e in eps if e['split']==split])
    split_manifest={'seed':42,'targets':str(args.targets.relative_to(ROOT)),
        'selection':'test demos0,12,24,36,49 previously fixed for zero-shot baseline; validation5/demo per task sampled seed42 from remaining; training40',
        'counts':{s:sum(e['split']==s for e in eps) for s in ['training','validation','test']},
        'pairs':{s:sum(e['pairs'] for e in eps if e['split']==s) for s in ['training','validation','test']},
        'episodes':eps}
    assert split_manifest['counts']=={'training':400,'validation':50,'test':50}
    write_json(args.output/'split_manifest.json',split_manifest)
    checkpoint=ROOT/'checkpoints/vera-jidm/idm-mimicgen-285ouq1q'
    cfg=yaml.safe_load((checkpoint/'config.yaml').read_text())
    # Released task-balanced validation entries point at MimicGen placeholders.
    # Route held-out evaluation solely to the complete LIBERO validation split.
    cfg.pop('validation_datasets',None)
    cfg.pop('validation_include_all',None)
    cfg.pop('_on_compute_node',None)
    cfg['skip_download']=True
    cfg.update(debug=False,name='libero_mimicgen_jidm_finetune',resume=None,load='${oc.env:PROJECT_ROOT}/checkpoints/vera-jidm/idm-mimicgen-285ouq1q/model.ckpt',
        wandb={'mode':'disabled','entity':'lf3r','project':'libero-jidm'})
    cfg['experiment']['_name']='jacobian_learning';cfg['algorithm']['_name']='image_jacobian'
    cfg['dataset']['_name']='robomimic'
    dataset_root=str(args.output.relative_to(ROOT))
    cfg['dataset']['root']='${oc.env:PROJECT_ROOT}/'+dataset_root+'/training'
    cfg['dataset']['data_root']=cfg['dataset']['root'];cfg['dataset']['seed']=42
    for split,key in [('validation','validation_dataset'),('test','test_dataset')]:
        cfg[key]=copy.deepcopy(cfg['dataset']);cfg[key]['root']='${oc.env:PROJECT_ROOT}/'+dataset_root+'/'+split;cfg[key]['data_root']=cfg[key]['root']
        cfg[key]['finite_eval_episodes']=True
    # Bundled VGGT weights have native 518px positional embeddings; runtime128px unchanged.
    cfg['algorithm']['model']['pretrained_model_id']=None;cfg['algorithm']['model']['image_size']=518
    cfg['experiment']['training'].update(batch_size=1,precision='bf16-mixed',override_optim_state=True,override_strict_load=True)
    cfg['experiment']['training']['optim']['accumulate_grad_batches']=16
    cfg['experiment']['training']['data'].update(num_workers=4,prefetch_factor=2)
    cfg['experiment']['validation']['data'].update(num_workers=2,prefetch_factor=2)
    cfg['experiment']['validation'].update(limit_batch=50,val_every_n_step=1000)
    cfg['hydra']={'run':{'dir':'${oc.env:PROJECT_ROOT}/outputs/vera-libero-training/${now:%Y%m%d_%H%M%S}'}}
    dest=args.output/'configs';dest.mkdir(exist_ok=True)
    (dest/'libero_idm.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    assert cfg['dataset']['action_abs_scale']==yaml.safe_load((checkpoint/'config.yaml').read_text())['dataset']['action_abs_scale']
    write_json(args.output/'preparation.json',{'status':'FLOW_PENDING','targets':str(args.targets.relative_to(ROOT)),
        'split_manifest_sha256':sha256(args.output/'split_manifest.json'),'checkpoint':'checkpoints/vera-jidm/idm-mimicgen-285ouq1q',
        'flow_backend':'megaflow-flow','flow_window':4,'flow_refinement_iterations':8,'flow_codec':'qint8_zstd_npz',
        'rgb_codec':'png lossless (official PIL decoder compatible)','flow_fix_width':True,'training_started':False})
    print('SPLITS',split_manifest['counts'],split_manifest['pairs'],flush=True)


if __name__=='__main__':main()
