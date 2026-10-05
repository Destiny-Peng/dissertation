"""Per-dimension oracle and trajectory-held-out affine diagnostic, not deployment."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from common import ROOT, DIMENSIONS, read_plan, write_json, metrics, sha256


def fit(x, y):
    centered = x - x.mean(0)
    den = (centered ** 2).sum(0)
    alpha = np.divide((centered * (y-y.mean(0))).sum(0), den, out=np.zeros(6), where=den>1e-12)
    return alpha, y.mean(0) - alpha * x.mean(0)


def diagnostics(x, y):
    rows=[]
    for i in range(6):
        a,b=x[:,i],y[:,i]
        active=np.abs(b)>0.01
        rows.append({'dimension':DIMENSIONS[i], 'pearson_r':float(np.corrcoef(a,b)[0,1]),
            'sign_agreement_all':float((np.sign(a)==np.sign(b)).mean()),
            'sign_agreement_gt_abs_over_0.01':float((np.sign(a[active])==np.sign(b[active])).mean()),
            'active_count':int(active.sum()), 'pred_mean':float(a.mean()),'gt_mean':float(b.mean()),
            'pred_std':float(a.std()),'gt_std':float(b.std()),
            'std_ratio':float(a.std()/b.std()),'pred_rms':float(np.sqrt((a*a).mean())),
            'gt_rms':float(np.sqrt((b*b).mean()))})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert args.output.resolve().is_relative_to(ROOT)
    plan=read_plan(args.source);entries=[]
    for task in plan['tasks']:
        for fold,demo in enumerate(task['demos']):
            p=np.load(args.source/demo['output']/'prediction.npz')
            entries.append((task,demo,fold,p))
    y=np.concatenate([e[3]['gt_action'][:,:6] for e in entries]);folds=np.concatenate([np.full(e[1]['pairs'],e[2]) for e in entries])
    summary={'timestamp':datetime.now().astimezone().isoformat(),'source':str(args.source.relative_to(ROOT)),
        'demos':50,'pairs':len(y),'protocol':'OLS six independent dimensions; oracle fit on all evaluation pairs; five folds each holds one complete demo per task; frame-weighted; unchanged predicted gripper',
        'sign_deadband_gt':0.01,'playback_selection':'demo_24 for each task; starts 0, midpoint, final 30 pairs; horizon30; raw/oracle/OOF unchanged gripper plus diagnostic OOF with GT gripper; no model rerun',
        'zero_arm':metrics(np.zeros_like(y),y),'variants':{},'source_prediction_hashes':{e[1]['id']:sha256(args.source/e[1]['output']/'prediction.npz') for e in entries}}
    for name,key in [('unclipped','predicted_action_unclipped'),('clipped','predicted_action')]:
        x=np.concatenate([e[3][key][:,:6] for e in entries]);alpha,beta=fit(x,y)
        oracle=np.clip(x*alpha+beta,-1,1);oof=np.empty_like(x);coeff=[]
        for fold in range(5):
            train=folds!=fold;test=~train;a,b=fit(x[train],y[train]);oof[test]=np.clip(x[test]*a+b,-1,1)
            coeff.append({'fold':fold,'heldout_demo_ids':[e[1]['id'] for e in entries if e[2]==fold],
                'train_pairs':int(train.sum()),'test_pairs':int(test.sum()),'alpha':a.tolist(),'beta':b.tolist()})
        summary['variants'][name]={'dimension_diagnostics':diagnostics(x,y),'alpha':alpha.tolist(),'beta':beta.tolist(),
            'raw_arm':metrics(np.clip(x,-1,1),y),'oracle_arm':metrics(oracle,y),'heldout_arm':metrics(oof,y),'fold_coefficients':coeff}
        offset=0
        for task,demo,fold,p in entries:
            n=demo['pairs'];folder=args.output/demo['output'];folder.mkdir(parents=True,exist_ok=True)
            def actions(arm):return np.column_stack([arm,p['predicted_action'][:,6]])
            np.savez_compressed(folder/(name+'_actions.npz'),raw=p['predicted_action'],oracle=actions(oracle[offset:offset+n]),heldout=actions(oof[offset:offset+n]),gt=p['gt_action'],pair_index=p['pair_index'])
            offset+=n
    write_json(args.output/'diagnostic.json',summary)
    print(json.dumps({k:{'r':[d['pearson_r'] for d in v['dimension_diagnostics']],'alpha':v['alpha'],'oracle_mae':v['oracle_arm']['mae'],'heldout_mae':v['heldout_arm']['mae']} for k,v in summary['variants'].items()},indent=2))


if __name__=='__main__':main()
