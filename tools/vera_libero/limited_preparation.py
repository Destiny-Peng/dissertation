"""Resume an explicit capped selection, without launching the training queue."""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def main():
    data = Path((ROOT / 'cache/libero_jidm_training_current_run.txt').read_text().strip())
    run = Path((ROOT / 'cache/libero_jidm_training_authorized_run.txt').read_text().strip())
    selection = data / 'selection_200.json'
    ids = set(json.loads(selection.read_text())['episode_ids'])
    manifest = json.loads((data / 'split_manifest.json').read_text())
    episodes = [ep for ep in manifest['episodes'] if ep['episode_id'] in ids]
    assert len(ids) == len(episodes) == 200
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                command = (proc / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if str(ROOT / 'tools/vera_libero/pack_training.py').encode() in command:
                raise RuntimeError('An existing pack worker is active; refusing duplicate launch')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workers = []
    record = {'pid': os.getpid(), 'status': 'RUNNING', 'limit': 200,
              'training_started': False,
              'training_after_preparation': os.environ.get('LF3R_TRAIN_AFTER_LIMIT200') == '1',
              'selection': str(selection.relative_to(ROOT)), 'workers': []}
    def publish(status, detail):
        record.update(status=status, detail=detail,
                      updated_at=datetime.now().astimezone().isoformat())
        record['completed_demos'] = sum((data / ep['path']).exists() and
                                       (data / ep['path']).with_suffix('.json').exists()
                                       for ep in episodes)
        write(data / 'worker_status.json', record)
        write(run / 'limited_preparation.json', record)
    write(run / 'queue_status.json', {'status': 'DISABLED_FOR_200_DEMO_PREPARATION',
                                     'training_started': False})
    try:
        for gpu, batch in [('0', '2'), ('1', '4'), ('2', '2')]:
            if gpu not in os.environ.get('LF3R_LIMIT200_GPUS', '0,1,2').split(','):
                continue
            log = ROOT / f'logs/libero_jidm_limit200_gpu{gpu}_{stamp}.log'
            command = [os.environ['LF3R_VERA_JIDM_PYTHON'],
                       str(ROOT / 'tools/vera_libero/pack_training.py'),
                       '--data', str(data), '--dynamic', '--auto-batch',
                       '--batch', batch, '--max-batch', '8',
                       '--shard-index', gpu, '--shard-count', '3',
                       '--episode-list', str(selection)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            with log.open('a') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
            workers.append(child)
            record['workers'].append({'gpu': gpu, 'pid': child.pid,
                                      'command': command, 'log': str(log.relative_to(ROOT))})
        publish('RUNNING', 'User-authorized restart; exactly 200 selected demos')
        while any(child.poll() is None for child in workers):
            if any(child.poll() not in (None, 0) for child in workers):
                raise RuntimeError('A packing worker failed; see per-GPU logs. No automatic retries.')
            publish('RUNNING', 'Packing only the remaining selected demos')
            time.sleep(5)
        if any(child.returncode != 0 for child in workers):
            raise RuntimeError('Packing worker failed')
        assert all((data / ep['path']).exists() and
                   (data / ep['path']).with_suffix('.json').exists() for ep in episodes)
        publish('STOPPED_AT_LIMIT', 'All 200 selected demos cached; preprocessing stopped; training not launched')
        if os.environ.get('LF3R_TRAIN_AFTER_LIMIT200') == '1':
            publish('VERIFYING_SUBSET', 'All 200 cached; verifying subset before authorized training')
            subprocess.run([os.environ['LF3R_VERA_JIDM_PYTHON'],
                            str(ROOT / 'tools/vera_libero/train_subset200.py'),
                            '--data', str(data), '--run', str(run)], cwd=ROOT, check=True)
    except BaseException as error:
        for child in workers:
            if child.poll() is None:
                child.terminate()
        for child in workers:
            child.wait()
        publish('BLOCKED', str(error))
        raise
    finally:
        with (ROOT / 'SETUP_STATUS.md').open('a') as stream:
            stream.write(f"\nLIBERO J-IDM capped restart {record['updated_at']}: "
                         f"{record['status']}, {record['completed_demos']}/200 cached. "
                         f"{record['detail']}. See {run.relative_to(ROOT)}/limited_preparation.json.\n")


if __name__ == '__main__':
    main()
