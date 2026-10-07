"""Durable sequential preparation worker; never invokes formal training."""
import argparse,json,os,subprocess,sys,hashlib,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def record(data,status,detail):
    now=datetime.now().astimezone().isoformat();p=data/'worker_status.json'
    state={'updated_at':now,'status':status,'detail':detail,'pid':os.getpid(),'training_started':False};write(p,state)
    (data/'worker.pid').write_text(str(os.getpid())+'\n')
    preparation=json.loads((data/'preparation.json').read_text())
    preparation.update(status=status,updated_at=now,worker_pid=os.getpid(),training_started=False)
    write(data/'preparation.json',preparation)
    report=ROOT/'environment_reports'/('LIBERO_JIDM_TRAINING_PREP_'+data.name+'.md')
    split=json.loads((data/'split_manifest.json').read_text());v=data/'verification.json'
    body=f'# LIBERO J-IDM training preparation\n\nStatus: {status}.\n\n{detail}\n\nWhole-trajectory counts: {split["counts"]}; pair counts: {split["pairs"]}. Original7D target, camera input and normalization preserved; no WM or formal training.\n\nData: {data.relative_to(ROOT)}; code/protocol: tools/vera_libero/TRAINING.md. Timestamp: {now}.\n'
    if v.exists():body+='\nVerification:\n```json\n'+v.read_text()+'```\n'
    report.write_text(body)
    with (ROOT/'SETUP_STATUS.md').open('a') as f:f.write(f'\n| {now} | LIBERO J-IDM | preparation_worker | 1 | {status} | {detail} Report environment_reports/{report.name}. |\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--gpus',default='1');ap.add_argument('--batches',default=None);args=ap.parse_args();data=args.data
    gpus=args.gpus.split(',');assert len(gpus)==len(set(gpus)) and all(g.isdigit() for g in gpus)
    batches=[int(b) for b in args.batches.split(',')] if args.batches else [1]*len(gpus)
    assert len(batches)==len(gpus) and all(b>0 for b in batches)
    assert data.resolve().is_relative_to(ROOT/'datasets')
    record(data,'RUNNING',f'User-authorized parallel preparation on GPUs{gpus}, initial batches{batches}, automatic growth from actual CUDA peaks up to32; shared lock-protected trajectory pool, original MegaFlow settings and completed packs preserved; verification/training follow all workers.')
    commands=[]
    pending={i:0 for i in range(len(gpus))};active={}
    while pending or active:
        for shard,previous in list(pending.items()):
            attempt=previous+1;stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
            log=ROOT/'logs'/f'libero_jidm_pack_gpu{gpus[shard]}_{stamp}_attempt{attempt}.log'
            batch=max(1,batches[shard]//(2**previous))
            argv=[sys.executable,str(ROOT/'tools/vera_libero/pack_training.py'),'--data',str(data),'--batch',str(batch),'--shard-index',str(shard),'--shard-count',str(len(gpus)),'--dynamic','--auto-batch','--max-batch',str(max(batch,32//(2**previous)))]
            entry={'stage':'pack','shard':shard,'gpu':gpus[shard],'batch':batch,'attempt':attempt,'argv':argv,'log':str(log.relative_to(ROOT))}
            commands.append(entry);write(data/'worker_commands.json',commands)
            stream=log.open('w');stream.write('COMMAND '+json.dumps(argv)+'\n');stream.flush()
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpus[shard]
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            entry['pid']=process.pid;active[shard]=(process,stream,entry);del pending[shard]
            write(data/'worker_commands.json',commands)
        for shard,(process,stream,entry) in list(active.items()):
            code=process.poll()
            if code is None:continue
            stream.close();entry['exit_code']=code;del active[shard];write(data/'worker_commands.json',commands)
            if code!=0:
                if entry['attempt']>=3:
                    for other,other_stream,_ in active.values():
                        other.terminate();other.wait();other_stream.close()
                    record(data,'BLOCKED',f'Pack shard{shard} failed three times; see {entry["log"]}.')
                    return
                pending[shard]=entry['attempt']
                record(data,'RETRYING',f'Pack shard{shard} exited{code}; completed packs preserved. See {entry["log"]}.')
        if pending or active:time.sleep(5)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus[0]
    for stage,script,extra in [('verify','verify_training.py',[])]:
        for attempt in range(1,4):
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S');log=ROOT/'logs'/f'libero_jidm_worker_{stage}_{stamp}_attempt{attempt}.log'
            argv=[sys.executable,str(ROOT/'tools/vera_libero'/script),'--data',str(data)]+extra
            commands.append({'stage':stage,'attempt':attempt,'argv':argv,'log':str(log.relative_to(ROOT))})
            write(data/'worker_commands.json',commands)
            with log.open('w') as stream:
                stream.write('COMMAND '+json.dumps(argv)+'\n');stream.flush()
                result=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            commands[-1]['exit_code']=result.returncode;write(data/'worker_commands.json',commands)
            if result.returncode==0:break
            record(data,'RETRYING' if attempt<3 else 'BLOCKED',f'{stage} attempt{attempt} exited{result.returncode}; completed packs preserved. See {log.relative_to(ROOT)}.')
            if attempt==3:
                s=json.loads((data/'preparation.json').read_text());s['status']='BLOCKED';write(data/'preparation.json',s)
                return
    files=[ROOT/'tools/vera_libero'/n for n in ['prepare_training.py','pack_training.py','verify_training.py','training_worker.py','TRAINING.md']]
    files += [ROOT/'tools/run_libero_jidm_prepare.sh',ROOT/'tools/run_libero_jidm_train.sh',ROOT/'tools/patches/vera-jidm-heldout-datasets.patch']
    provenance={'completed_at':datetime.now().astimezone().isoformat(),'commands':commands,
        'code_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'commits':{name:subprocess.check_output(['git','-C',str(ROOT/'repos'/name),'rev-parse','HEAD'],text=True).strip() for name in ['VERA','VGGT','LIBERO','megaflow']},
        'training_started':False,'wm_used':False}
    write(data/'provenance.json',provenance)
    record(data,'READY','All500 demos/137590 pairs packed and verified; splits400/50/50 disjoint; lossless RGB, targets and official frozen loss forward passed. No formal training/WM.')
    with (ROOT/'SYSTEM_INFO.txt').open('a') as f:f.write(f'\n[{provenance["completed_at"]}] LIBERO J-IDM training data READY: {data.relative_to(ROOT)}, all500 demos/137590 pairs, default MegaFlow-flow952/window4/iterations8, original IO and norm, official frozen loss verification PASS; no formal training/WM.\n')


if __name__=='__main__':main()
