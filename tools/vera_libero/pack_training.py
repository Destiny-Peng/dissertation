"""Official-compatible LIBERO packed episodes with frozen MegaFlow supervision."""
import argparse,json,sys,time,fcntl,os
from pathlib import Path
import h5py,numpy as np,torch
from safetensors.torch import load_file
from common import ROOT,write_json,sha256
sys.path.insert(0,str(ROOT/'repos/megaflow'))
from megaflow.model import MegaFlow
from vera.utils.droid_packed_format import encode_image_bytes,encode_flow_payload,decode_flow_payload,write_npz_stored_atomic,metadata_to_uint8_array
from vera.datasets.core.packed import packed_entry_key,rgb_entry_key,flow_entry_key,PACKED_METADATA_KEY


class BatchController:
    def __init__(self, args):
        self.args=args;self.batch=args.batch;self.per_window_peak=None;self.observed_batches=set()
        self.control=args.data/'batch_control.json'
        self.record=args.data/f'batch_runtime.gpu{os.environ.get("CUDA_VISIBLE_DEVICES",args.shard_index)}.json'

    def next_batch(self):
        config=json.loads(self.control.read_text()) if self.control.exists() else {}
        cap=min(self.args.max_batch,int(config.get('max_batch',self.args.max_batch)))
        cap=int(config.get('gpu_limits',{}).get(os.environ.get('CUDA_VISIBLE_DEVICES',''),cap))
        cap=min(cap,self.args.max_batch)
        margin=float(config.get('headroom_gib',2))*1024**3
        if self.per_window_peak is not None:
            free,_=torch.cuda.mem_get_info()
            available=free+torch.cuda.memory_reserved()-torch.cuda.memory_allocated()-margin
            capacity=max(1,int(available/(self.per_window_peak*1.15)))
            target=max(1,min(cap,capacity,self.batch*2))
        else:target=max(1,min(self.batch,cap))
        if target!=self.batch:
            print('AUTO_BATCH_CHANGE',self.batch,'->',target,flush=True);self.batch=target
        return self.batch

    def observe(self, size, full_window, baseline, seconds):
        peak=torch.cuda.max_memory_allocated()
        if full_window:
            incremental=max((peak-baseline)/size,1024**2)
            self.per_window_peak=max(self.per_window_peak or 0,incremental)
        free,_=torch.cuda.mem_get_info();self.observed_batches.add(size)
        state={'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'actual_batch':size,
               'next_batch_seed':self.batch,'observed_batches':sorted(self.observed_batches),
               'peak_allocated_gib':peak/1024**3,'reserved_gib':torch.cuda.memory_reserved()/1024**3,
               'global_free_gib':free/1024**3,'incremental_peak_gib_per_window':(self.per_window_peak or 0)/1024**3,
               'seconds_per_window':seconds/size,'max_batch':self.args.max_batch,
               'updated_at':__import__('datetime').datetime.now().astimezone().isoformat()}
        write_json(self.record,state)
        print('BATCH_MEMORY',json.dumps(state),flush=True)


@torch.inference_mode()
def flows(model, rgb, chunk_batch, controller=None):
    n=len(rgb);out=[];began=time.time()
    starts=list(range(0,n-1,3))
    # Same 4-frame/one-frame overlap as official VERA packer; batch only equal-sized windows.
    offset=0;iteration=0
    while offset<len(starts):
        chunk_batch=controller.next_batch() if controller else chunk_batch
        group=starts[offset:offset+chunk_batch];lengths=[min(4,n-s) for s in group]
        if len(set(lengths))>1:
            groups=[group[:-1],group[-1:]]
        else:groups=[group]
        for subset in groups:
            if not subset:continue
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();step_began=time.time()
            video=np.stack([rgb[s:min(s+4,n)] for s in subset])
            x=torch.from_numpy(video).permute(0,1,4,2,3).float().to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):result=model(x,num_reg_refine=8)['flow_preds'][-1]
            result=result.float().cpu().numpy()
            assert result.shape[0]==len(subset) and result.shape[2:]==(2,128,128)
            assert np.isfinite(result).all()
            out.extend(result)
            del x
            if controller:controller.observe(len(subset),video.shape[1]==4,baseline,time.time()-step_began)
        if iteration%5==0:
            print('FLOW_WINDOWS',offset,'/',len(starts),'seconds',round(time.time()-began,1),flush=True)
        offset+=len(group);iteration+=1
    flow=np.concatenate(out,axis=0);assert flow.shape==(n-1,2,128,128)
    return flow


def claim_episodes(args, manifest):
    """Hold an OS lock through each yield so workers share unfinished demos."""
    lock_dir=args.data/'episode_locks';lock_dir.mkdir(exist_ok=True)
    selected = set(json.loads(args.episode_list.read_text())['episode_ids']) if args.episode_list else None
    for episode_index,ep in enumerate(manifest['episodes']):
        if selected is not None and ep['episode_id'] not in selected:continue
        if not args.dynamic and episode_index%args.shard_count!=args.shard_index:continue
        dest=args.data/ep['path'];report=dest.with_suffix('.json')
        if dest.exists() and report.exists():continue
        with (lock_dir/(dest.stem+'.lock')).open('a') as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            try:
                if dest.exists() and report.exists():continue
                yield ep
            finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--batch',type=int,default=1);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--shard-count',type=int,default=1)
    ap.add_argument('--dynamic',action='store_true')
    ap.add_argument('--episode-list',type=Path,help='Restrict packing to an explicit shared episode allowlist')
    ap.add_argument('--auto-batch',action='store_true');ap.add_argument('--max-batch',type=int,default=32)
    args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    assert args.shard_count>0 and 0<=args.shard_index<args.shard_count
    manifest=json.loads((args.data/'split_manifest.json').read_text());targets=ROOT/manifest['targets']
    checkpoint=ROOT/'checkpoints/vera-jidm/megaflow/megaflow-flow.safetensors';release=json.loads(checkpoint.with_name('release_file.json').read_text())
    checksum=sha256(checkpoint);assert checksum==release['lfs']['oid']
    model=MegaFlow(fuse_cnn=True,use_temporal_attn=True,fix_width=True)
    sd=load_file(str(checkpoint),device='cpu');model.load_state_dict(sd,strict=True);del sd
    model.eval().requires_grad_(False).cuda()
    controller=BatchController(args) if args.auto_batch else None
    model_record='flow_model.json' if args.shard_count==1 else f'flow_model.shard_{args.shard_index}.json'
    write_json(args.data/model_record,{'checkpoint_sha256':checksum,'strict_load':True,'model':'megaflow-flow','fix_width':True,'internal_feature_width':952,'window':4,'refinement_iterations':8,'dtype':'bf16 autocast; output float32','batch':args.batch,'auto_batch':args.auto_batch,'max_batch':args.max_batch,'shard_index':args.shard_index,'shard_count':args.shard_count,'pid':os.getpid()})
    count=0
    for ep in claim_episodes(args, manifest):
        dest=args.data/ep['path'];report=dest.with_suffix('.json')
        if dest.exists() and report.exists():continue
        import shutil
        assert shutil.disk_usage(args.data).free > 10*1024**3, 'Less than10GiB free; pause packing before exhausting disk'
        began=time.time();label=np.load(targets/ep['target_npz']);n=ep['frames'];views=['agentview_image','robot0_eye_in_hand_image']
        metadata={'num_frames':n,'episode_id':ep['episode_id'],'source_relative_path':ep['source_hdf5'],'views':views,'rgb_entries':{},'flow_entries':{},'trajectory_entries':{}}
        validation={'flow_finite':True,'max_quantization_error':0.,'max_quantization_step':0.}
        def entries():
            for key,arr in [('low_dim/robot0_eef_pos',label['ee_pos']),('low_dim/robot0_eef_quat',label['ee_quat']),('low_dim/robot0_gripper_qpos',label['gripper_states'])]:
                # Preserve original float64 source states for exact target parity.
                metadata['trajectory_entries'][key]={'shape':list(arr.shape),'dtype':str(arr.dtype)}
                yield packed_entry_key(key),arr
            with h5py.File(ROOT/ep['source_hdf5'],'r') as f:
                g=f['data/'+ep['demo']]
                for view,source in zip(views,['agentview_rgb','eye_in_hand_rgb']):
                    rgb=g['obs/'+source][:]
                    metadata['rgb_entries'][view]={'codec':'png','num_frames':n,'frame_shape':[128,128,3]}
                    for i,frame in enumerate(rgb):yield rgb_entry_key(i,view),np.frombuffer(encode_image_bytes(frame,codec='png'),dtype=np.uint8)
                    flow=flows(model,rgb,args.batch,controller)
                    metadata['flow_entries'][view]={'backend':'megaflow-flow','codec':'qint8_zstd_npz','dtype':'float32','num_frames':n-1,'frame_shape':[2,128,128],'source_frame_shape':[2,128,128]}
                    for i,frame in enumerate(flow):
                        payload=encode_flow_payload(frame,codec='qint8_zstd_npz');decoded=decode_flow_payload(payload,codec='qint8_zstd_npz')
                        error=float(np.abs(frame-decoded).max());step=max(float(np.abs(frame).max())/127,1e-6)
                        assert error<=step*.5001+1e-5
                        validation['max_quantization_error']=max(validation['max_quantization_error'],error)
                        validation['max_quantization_step']=max(validation['max_quantization_step'],step)
                        yield flow_entry_key(i,view),np.frombuffer(payload,dtype=np.uint8)
                    print('VIEW_DONE',ep['episode_id'],view,'seconds',round(time.time()-began,1),flush=True)
            yield PACKED_METADATA_KEY,metadata_to_uint8_array(metadata)
        write_npz_stored_atomic(dest,entries())
        write_json(report,{'episode_id':ep['episode_id'],'split':ep['split'],'pairs':ep['pairs'],'packed_sha256':sha256(dest),'bytes':dest.stat().st_size,'seconds':time.time()-began,'flow_batch':'auto' if controller else args.batch,'observed_batches':sorted(controller.observed_batches) if controller else [args.batch],'worker_index':args.shard_index,**validation})
        count+=1;print('PACK_DONE',ep['episode_id'],'seconds',round(time.time()-began,1),flush=True)
        if args.limit and count>=args.limit:break
    print('PACK_STAGE_FINISHED',count,flush=True)


if __name__=='__main__':main()
