"""Fixed short-horizon paired simulator diagnostic; calibration is not zero shot."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT,read_plan,write_json
from playback import build_env,restore,pose,errors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();plan=read_plan(args.source);records=[]
    modes=['gt','raw','oracle','heldout','heldout_gt_gripper']
    for task in plan['tasks']:
        demo=task['demos'][2];folder=args.output/demo['output'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'short_playback.json').exists():
            records.extend(json.loads((folder/'short_playback.json').read_text()));continue
        ref=np.load(args.source/demo['output']/'reference.npz');actions=np.load(folder/'unclipped_actions.npz')
        with h5py.File(ROOT/task['file'],'r') as f:
            xml=f['data/'+demo['demo']].attrs['model_file']
            if isinstance(xml,bytes):xml=xml.decode()
        env=build_env(task,xml);local=[]
        try:
            for start in sorted(set([0,demo['pairs']//2,demo['pairs']-30])):
                trajectories={};saved={}
                for mode in modes:
                    restore(env,ref['states'][start+1]);positions=[];rotations=[];grips=[];states=[];err=[]
                    selected=actions['heldout' if mode=='heldout_gt_gripper' else mode][start:start+30].copy()
                    if mode=='heldout_gt_gripper':selected[:,6]=actions['gt'][start:start+30,6]
                    for k,action in enumerate(selected):
                        obs,_,_,_=env.step(action);p,r,g=pose(obs)
                        positions.append(p);rotations.append(r.as_rotvec());grips.append(g);states.append(env.get_sim_state());err.append(errors(obs,ref,start+k+1))
                    trajectories[mode]=(np.asarray(positions),np.asarray(rotations),np.asarray(grips),np.asarray(err))
                    saved.update({mode+'_actions':selected,mode+'_states':np.asarray(states),mode+'_positions':positions,mode+'_rotations':rotations,mode+'_gripper':grips,mode+'_recorded_pose_error':err})
                for mode in modes:
                    p,r,g,err=trajectories[mode];gp,gr,gg,_=trajectories['gt']
                    delta=np.linalg.norm(p-gp,axis=1);rot=(Rotation.from_rotvec(r)*Rotation.from_rotvec(gr).inv()).magnitude()
                    row={'task':task['task_name'],'demo':demo['demo'],'start_pair':start,'steps':len(p),'mode':mode,
                        'mean_position_vs_recorded_m':float(err[:,0].mean()),'mean_position_vs_gt_playback_m':float(delta.mean()),
                        'final_position_vs_gt_playback_m':float(delta[-1]),'mean_rotation_vs_gt_playback_rad':float(rot.mean()),
                        'position_vs_gt_at_10_m':float(delta[9]),'position_vs_gt_at_30_m':float(delta[29])}
                    local.append(row)
                np.savez_compressed(folder/('segment_'+str(start)+'.npz'),**saved)
            write_json(folder/'short_playback.json',local);records.extend(local)
            print('SHORT_PLAYBACK_DONE',task['task_name'],flush=True)
        finally:env.close()
    summary={'tasks':10,'demos':10,'segments':len(records)//len(modes),'horizon_steps':30,'control_hz':20,
        'primary_variant':'unclipped action input to affine; clipping only after affine',
        'protocol':'paired open-loop 1.5s segments restored from same state; oracle coefficients use evaluation GT; heldout coefficients exclude entire demo; heldout_gt_gripper is explicitly oracle gripper diagnostic; no success-rate claim for short segments',
        'metrics':{mode:{key:float(np.mean([r[key] for r in records if r['mode']==mode])) for key in records[0] if key.startswith(('mean_','final_','position_'))} for mode in modes},'segments_detail':records}
    write_json(args.output/'short_playback_summary.json',summary);print(json.dumps(summary['metrics'],indent=2),flush=True)


if __name__=='__main__':main()
