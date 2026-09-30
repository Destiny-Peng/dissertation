#!/usr/bin/env python3
"""Build and review task-matched external robot evaluation pairs.

Uses cached official indexes under cache/matched_pair_sources. Candidate output
is deliberately separate from the reviewed evaluation manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import urllib.parse
import urllib.request
import zipfile
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'cache/matched_pair_sources'
OUT = ROOT / 'datasets/external_failure_success_pairs/v1'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def relative(path):
    return str(path.relative_to(ROOT))


def slug(value):
    return re.sub(r'[^A-Za-z0-9._-]', '_', value)


def selected(dataset):
    return read(ROOT / f'datasets/{dataset}_subset/selected_episodes.json')['episodes']


def base(dataset, failure, success_id, evidence):
    pid = dataset + '-' + hashlib.sha256(failure['episode_id'].encode()).hexdigest()[:12]
    return dict(pair_id=pid, dataset=dataset, task=failure['task'],
                failure_episode=failure['episode_id'], success_episode=success_id,
                failure_video=relative(ROOT / f'datasets/{dataset}_subset' / failure['video_path']),
                success_video=relative(OUT / dataset / 'success_videos' / (slug(success_id) + '.mp4')),
                goal_image=relative(OUT / dataset / 'goal_images' / (pid + '.png')),
                camera_view=failure['camera_view'], match_evidence=evidence,
                selection_status='candidate_requires_visual_review')


def droid_candidates():
    annotations = read(ROOT / 'cache/droid_annotations/droid_language_annotations.json')
    records, rejected = [], []
    matches = read(CACHE / 'droid_initial_matches.json')
    replacements = CACHE / 'droid_replacement_matches.json'
    if replacements.exists():
        matches += read(replacements)
    for m in matches:
        f = m['failure']; fm = m['failure_metadata']
        if f['task'].startswith(('Do any', 'Do anything')) or not m['success_candidates']:
            rejected.append(dict(failure_episode=f['episode_id'], reason='open-ended task or no same-task same-scene success'))
            continue
        candidates = [s for s in m['success_candidates'] if s['metadata'].get('ext1_cam_serial') == fm.get('ext1_cam_serial')]
        if not candidates:
            rejected.append(dict(failure_episode=f['episode_id'], reason='no matching camera serial'))
            continue
        s = candidates[0]; sm = s['metadata']
        fe, se = fm.get('ext1_cam_extrinsics'), sm.get('ext1_cam_extrinsics')
        delta = max(abs(a-b) for a,b in zip(fe,se)) if fe and se else None
        e = dict(task_match='exact normalized current_task', same_scene=True,
                 scene_id=fm['scene_id'], building=fm['building'], lab=fm['lab'],
                 same_date=fm['date'] == sm['date'], same_camera_serial=True,
                 camera_serial=fm['ext1_cam_serial'], max_camera_extrinsics_delta=delta,
                 failure_success_flag=False, success_success_flag=True,
                 failure_metadata_object=f['source']['metadata_object'],
                 success_metadata_object=s['object_name'],
                 success_language_annotations=annotations.get(sm['uuid']))
        r = base('droid', f, sm['uuid'], e)
        obj = f"robotics/droid_raw/1.0.1/{sm['lab']}/{sm['ext1_mp4_path']}"
        r['success_url'] = 'https://storage.googleapis.com/gresearch/' + urllib.parse.quote(obj, safe='/')
        r['failure_label_kind'] = 'official_terminal_failure'
        r['success_label_kind'] = 'official_success_true'
        r['failure_source_record'] = f
        records.append(r)
    return records, rejected


def robovad_candidates():
    import download_robovad_subset as rv
    index = read(CACHE / 'robovad_zip_index.json')
    entries = {e['name']: e for e in index['entries']}
    annotations = read(CACHE / 'robovad_annotations.json')
    split = {}
    for s in ('train','validation','test'):
        with (CACHE / f'robovad_{s}.csv').open() as stream:
            split.update({r['episode_name']:s for r in csv.DictReader(stream)})
    normals = [x for x in split if x not in annotations and f'RoboVAD/videos/{x}_left.mp4' in entries]
    records = []; used = set()
    for f in selected('robovad'):
        ns = [n for n in normals if rv.task_for(n) == f['task']]
        ns.sort(key=lambda n:(n in used, abs(rv.episode_index(n)-f['episode_index'])))
        if not ns: continue
        n=ns[0]; used.add(n)
        r = base('robovad', f, n, dict(task_match='exact episode-name task prefix',
                 failure_anomaly_events=annotations[f['episode_id']], success_anomaly_events=[],
                 success_split=split[n], same_camera_view=True,
                 scene_match='shared static setup; no episode scene ID provided',
                 success_basis='episode omitted from exhaustive anomaly annotations; visual endpoint review required'))
        r.update(success_archive_member=f'RoboVAD/videos/{n}_left.mp4',
                 failure_label_kind='official_anomaly_intervals_terminal_outcome_not_provided',
                 success_label_kind='official_normal_episode_plus_visual_review',
                 failure_source_record=f)
        records.append(r)
    return records, []


EXPERTS = {
    'han10e_recovery_install':'HAN10e-install', 'han10e_recovery_remove':'han10e_remove',
    'm12_recovery_install':'M12-fastener_install', 'm12_recovery_remove':'M12-fastener_remove',
    '16mm-cylinder_recovery_install':'16mm-cylinder_install',
    '16mm-cylinder_recovery_remove':'16mm-cylinder_remove',
    '16mm-bar_recovery_install':'16mm-bar_install', '16mm-bar_recovery_remove':'16mm-bar_remove',
    'rca_recovery_install':'rca_install', 'USBC_recovery_remove':'usb-c_remove',
    'nema1-15-plug_recovery_install':'nema1-15-plug_install',
    'nema1-15-plug_recovery_remove2':'nema1-15-plug_remove',
    'rj45_recovery_install':'rj45-install', 'RJ45_recovery_remove':'rj45_remove',
    'USB-A_recovery_install':'usbA-install',
}


def reboot_candidates():
    records=[]
    column='videos/observation.images.cam_high'
    for f in selected('reboot'):
        name=EXPERTS[f['source']['recovery_collection']]
        folder=CACHE/'reboot'/name; tables=read(folder/'tables.json')
        rows=[r for k,v in tables.items() if k.startswith('meta/episodes/') for r in v['rows']]
        # Episode index is a proximity tie-breaker, not an official pair identifier.
        rows.sort(key=lambda r:(' '.join(r['tasks']) != f['task'],abs(r['episode_index']-f['episode_index'])))
        s=rows[0]; rev=(folder/'revision.txt').read_text()
        shard=f"{column}/chunk-{s[column+'/chunk_index']:03d}/file-{s[column+'/file_index']:03d}.mp4"
        repo='REBOOT26/'+name; sid=f"{repo}#episode-{s['episode_index']}"
        tree=read(folder/'tree.json');file=next(t for t in tree if t['path']==shard)
        r=base('reboot',f,sid,dict(task_match='same assembly object and install/remove direction',
               failure_task_text=f['task'],success_task_text=s['tasks'],object=f['scene']['object'],
               direction=f['scene']['direction'],same_camera_view=True,
               scene_match='same published assembly station; visual comparison required',
               failure_collection='recovery-from-failure; NOT a terminal failure label',
               success_collection='expert demonstration',official_episode_pair_id=False))
        r.update(success_url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{shard}?download=true',
                 success_repo=repo,success_revision=rev,success_shard=shard,
                 success_shard_size=int(file.get('lfs',{}).get('size',file['size'])),
                 success_shard_sha256=file.get('lfs',{}).get('oid'),
                 success_start=s[column+'/from_timestamp'],success_end=s[column+'/to_timestamp'],
                 success_expected_frames=s['length'],failure_source_record=f,
                 failure_label_kind='recovery_from_induced_failure_terminal_outcome_unverified',
                 success_label_kind='expert_collection_plus_visual_review')
        records.append(r)
    return records,[]


def discover(dataset):
    records,rejected=globals()[dataset+'_candidates']()
    write(OUT/dataset/'candidates.json',records)
    write(OUT/dataset/'excluded_initial_failures.json',rejected)
    print(dataset,len(records),'candidates',len(rejected),'excluded',flush=True)


def probe(path):
    p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames:format=duration',
        '-of','json',str(path)],capture_output=True,text=True,check=True)
    d=json.loads(p.stdout);s=d['streams'][0]
    numerator,denominator=map(int,s['avg_frame_rate'].split('/'))
    return dict(width=s['width'],height=s['height'],fps=numerator/denominator,
                frames=int(s['nb_read_frames']),duration=float(d['format']['duration']))


def frames_and_review(record):
    import cv2
    from PIL import Image,ImageDraw
    from textwrap import wrap
    goal=ROOT/record['goal_image'];goal.parent.mkdir(parents=True,exist_ok=True)
    review=OUT/record['dataset']/'review'/f"{record['pair_id']}.jpg";review.parent.mkdir(exist_ok=True)
    images=[]
    for role,field in [('failure','failure_video'),('success','success_video')]:
        p=ROOT/record[field]; info=probe(p);record[role+'_video_info']=info
        cap=cv2.VideoCapture(str(p))
        for frac in (0,.35,.7,1):
            idx=min(info['frames']-1,round((info['frames']-1)*frac))
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {p} frame {idx}')
            if role=='success' and frac==1:
                # Confirm EOF after the last counted decoded frame.
                extra_ok,_=cap.read()
                if extra_ok:raise RuntimeError(f'Frame count inconsistent at EOF: {p}')
                if not cv2.imwrite(str(goal),frame):raise RuntimeError('Goal PNG write failed')
                record['goal_frame_index']=idx
                record['goal_timestamp_seconds']=idx/info['fps']
                record['goal_extraction']='last decoded frame of complete success episode; PNG without resizing'
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((360,240))
            images.append((role,idx,im))
        cap.release()
    sheet=Image.new('RGB',(1440,650),'#eeeeee');draw=ImageDraw.Draw(sheet)
    draw.text((8,5),record['pair_id']+' | '+record['task'][:180],fill='black')
    for j,(role,idx,im) in enumerate(images):
        x=(j%4)*360;y=45+(j//4)*285
        sheet.paste(im,(x,y+20));draw.text((x+5,y),f'{role} frame {idx}',fill='black')
    draw.text((8,625),record['success_episode'][:160],fill='black')
    sheet.save(review)
    record['review_sheet']=relative(review)


def materialize(dataset, pair_ids=None):
    records=read(OUT/dataset/'candidates.json')
    if dataset=='robovad':
        import download_robovad_subset as rv
        index=read(CACHE/'robovad_zip_index.json');reader=rv.RangeReader(rv.ARCHIVE_URL,index['size'])
        entries={e['name']:e for e in index['entries']}
    for n,r in enumerate(records,1):
        if pair_ids and r['pair_id'] not in pair_ids:
            continue
        failure_path = ROOT / r['failure_video']
        if dataset == 'droid' and not failure_path.exists():
            import download_droid_subset as dd
            f = r['failure_source_record']
            meta = read(CACHE / 'droid_metadata' / Path(f['source']['metadata_object']).name)
            obj = f"robotics/droid_raw/1.0.1/{meta['lab']}/{meta['ext1_mp4_path']}"
            url = 'https://storage.googleapis.com/gresearch/' + urllib.parse.quote(obj, safe='/')
            size = dd.head_size(url)
            dd.download_resumable(url, failure_path, size)
        p=ROOT/r['success_video'];p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and (ROOT/r['goal_image']).exists() and 'goal_frame_index' in r:
            print(dataset,n,len(records),r['pair_id'],'reused verified material',flush=True)
            continue
        if not p.exists():
            if dataset=='droid':
                import download_droid_subset as dd
                size=dd.head_size(r['success_url']);dd.download_resumable(r['success_url'],p,size)
            elif dataset=='robovad':
                e=entries[r['success_archive_member']];i=zipfile.ZipInfo(e['name'])
                i.file_size=e['size'];i.compress_size=e['compressed'];i.header_offset=e['offset'];i.CRC=e['crc'];i.compress_type=e['compression']
                staging=OUT/dataset/'.staging'/f"{r['pair_id']}.zipmember.part"
                bounded_zip_member(reader,i,staging)
                rv.extract_verified(i,staging,p);staging.unlink()
            else:
                import download_reboot_subset as rb
                shard=OUT/dataset/'.staging'/(slug(r['success_repo'])+'__'+Path(r['success_shard']).name)
                rb.download_resumable(r['success_url'],shard,r['success_shard_size'],r['success_shard_sha256'])
                make_frame_clip(shard,p,round(r['success_start']*30),r['success_expected_frames'])
                # Preserve source shard for candidate replacements and reproducibility.
                r['local_source_shard']=relative(shard)
        frames_and_review(r)
        write(OUT/dataset/'candidates.json',records)
        print(dataset,n,len(records),r['pair_id'],r['task'],flush=True)


def bounded_zip_member(reader, info, path):
    """Resume the compressed member with small verified HTTP ranges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header=reader.read_at(info.header_offset,30)
    if header[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP local header')
    name_size,extra_size=struct.unpack_from('<HH',header,26)
    start=info.header_offset+30+name_size+extra_size
    offset=path.stat().st_size if path.exists() else 0
    if offset>info.compress_size:raise RuntimeError('Oversized ZIP partial')
    blocks=[(start+i,min(512*1024,info.compress_size-i))
            for i in range(offset,info.compress_size,512*1024)]
    with path.open('ab') as stream,ThreadPoolExecutor(max_workers=6) as executor:
        for block in executor.map(lambda b:reader._fetch(*b),blocks):
            stream.write(block);stream.flush()
    if path.stat().st_size!=info.compress_size:raise RuntimeError('Short ZIP partial')


def make_frame_clip(source, destination, start_frame, count, fps=30):
    """Trim the exact source frame interval; avoid timestamp rounding at EOF."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.stem+'.exact.part.mp4')
    cmd=['ffmpeg','-v','error','-nostdin','-y','-threads','2','-i',str(source),
         '-map','0:v:0','-an','-vf',f'trim=start_frame={start_frame}:end_frame={start_frame+count},setpts=PTS-STARTPTS',
         '-frames:v',str(count),'-fps_mode','passthrough','-c:v','libx264','-threads','2',
         '-preset','fast','-crf','18','-movflags','+faststart',str(temp)]
    subprocess.run(cmd,check=True,capture_output=True)
    info=probe(temp)
    if info['frames']!=count:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f'Wrong episode frame count: {info["frames"]} != {count}')
    temp.replace(destination)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    temp.replace(path)


def manifest_pair(record):
    """Keep failure inputs and the goal path; store success provenance separately."""
    fields = ('pair_id', 'dataset', 'task', 'task_original', 'failure_episode',
              'failure_video', 'camera_view', 'failure_video_info',
              'failure_label_kind', 'terminal_failure_verified', 'evaluation_scope')
    result = {key: record[key] for key in fields if key in record}
    result['goal_image_path'] = record['goal_image']
    return result


def finalize(dataset):
    """Export only explicitly reviewed candidates, with truthful label scope."""
    import cv2
    import numpy as np
    candidates=read(OUT/dataset/'candidates.json')
    decisions=read(OUT/dataset/'review_decisions.json')
    retained=[];excluded=[]
    for r in candidates:
        d=decisions.get(r['pair_id'])
        if not d or not d.get('accepted'):
            excluded.append(dict(pair_id=r['pair_id'],failure_episode=r['failure_episode'],
                                 reason=(d or {}).get('reason','not approved after visual review')))
            continue
        for field in ('failure_video','success_video','goal_image'):
            if not (ROOT/r[field]).is_file():raise FileNotFoundError(r[field])
        if r['goal_frame_index']!=r['success_video_info']['frames']-1:
            raise RuntimeError('Goal does not refer to success final frame')
        cap=cv2.VideoCapture(str(ROOT/r['success_video']))
        cap.set(cv2.CAP_PROP_POS_FRAMES,r['goal_frame_index']);ok,frame=cap.read();cap.release()
        goal=cv2.imread(str(ROOT/r['goal_image']))
        if not ok or not np.array_equal(frame,goal):raise RuntimeError('Goal PNG differs from last decoded success frame')
        r=dict(r);r['task_original']=r['task'];r['task']=d.get('task',r['task'])
        r['visual_review']=d;r['selection_status']='accepted'
        r['goal_png_sha256']=hashlib.sha256((ROOT/r['goal_image']).read_bytes()).hexdigest()
        r['terminal_failure_verified']=dataset=='droid'
        r['evaluation_scope']='terminal failure and progress' if dataset=='droid' else 'failure-containing trajectory progress; no assumed terminal failure label'
        retained.append(r)
    if not 10<=len(retained)<=20:raise RuntimeError(f'{dataset}: expected 10–20 reviewed pairs, got {len(retained)}')
    if len({r['failure_episode'] for r in retained})!=len(retained):raise RuntimeError('Duplicate failure')
    write(OUT/dataset/'pairing_provenance.json',retained)
    write_jsonl(OUT/dataset/'pairing_manifest.jsonl',[manifest_pair(r) for r in retained])
    write(OUT/dataset/'excluded_candidates.json',excluded)
    model_records=[]
    camera={'droid':'ext1','robovad':'left','reboot':'cam_high'}[dataset]
    name={'droid':'DROID','robovad':'RoboVAD','reboot':'REBOOT'}[dataset]
    tasks=sorted({r['task'] for r in retained})
    for r in retained:
        info=r['failure_video_info']
        model_records.append(dict(schema_version=1,id=r['pair_id'],pair_id=r['pair_id'],
            source_dataset=name,task_suite=dataset,task_suite_label=name,
            task_id=tasks.index(r['task']),task_description=r['task'],task_description_original=r['task_original'],
            dataset_role=dataset+'_matched_failure_success',analysis_partition='real_robot_analysis',
            source_kind='external_dataset',source_episode_id=r['failure_episode'],episode_index=r['failure_source_record'].get('episode_index'),
            video_path=r['failure_video'],failure_video=r['failure_video'],camera_video_paths={camera:r['failure_video']},
            camera_view=r['camera_view'],goal_image_path=r['goal_image'],
            fps=info['fps'],duration_seconds=info['duration'],total_frames=info['frames'],csv_path=None,
            ground_truth_outcome='failure' if dataset=='droid' else 'unknown',contains_failure=True,
            failure_label_kind=r['failure_label_kind'],terminal_failure_verified=r['terminal_failure_verified'],
            evaluation_scope=r['evaluation_scope'],source=r['failure_source_record'].get('source',{})))
    write_jsonl(OUT/dataset/'robodopamine_manifest.jsonl',model_records)
    completed=[];combined=[];provenance=[]
    for ds in ('droid','robovad','reboot'):
        pp=OUT/ds/'pairing_manifest.jsonl';mp=OUT/ds/'robodopamine_manifest.jsonl'
        if pp.exists() and mp.exists():
            completed.extend(json.loads(line) for line in pp.read_text().splitlines())
            provenance.extend(read(OUT/ds/'pairing_provenance.json'))
            combined.extend(json.loads(line) for line in mp.read_text().splitlines())
    write_jsonl(OUT/'pairing_manifest.jsonl',completed)
    write_jsonl(OUT/'robodopamine_manifest.jsonl',combined)
    summary=dict(generated_at=datetime.now(timezone.utc).isoformat(),counts=dict(Counter(r['dataset'] for r in completed)),
        total_pairs=len(completed),unique_success_rollouts=len({(r['dataset'],r['success_episode']) for r in provenance}),
        terminal_failure_label_policy='DROID: official success=false; RoboVAD: anomalous intervals; REBOOT: recovery-from-failure. Only DROID is asserted terminal failure.',
        goal_policy='unresized lossless PNG of last decoded frame of complete successful episode',
        completed_datasets=sorted({r['dataset'] for r in completed}))
    write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['discover','materialize','finalize'])
    parser.add_argument('--dataset',choices=['droid','robovad','reboot'],required=True)
    parser.add_argument('--pair-id',action='append',default=[])
    args=parser.parse_args()
    if args.phase=='materialize':materialize(args.dataset,set(args.pair_id))
    else:globals()[args.phase](args.dataset)


if __name__=='__main__':main()
