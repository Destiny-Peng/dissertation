"""Detached preparation and authorized training chain; standard library only."""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--batches', default=None)
    args = parser.parse_args()
    assert args.data.resolve().is_relative_to(ROOT / 'datasets')
    assert args.run.resolve().is_relative_to(ROOT / 'outputs')
    python = os.environ['LF3R_VERA_JIDM_PYTHON']
    (args.run / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in [args.data / 'worker_status.json', args.run / 'queue_status.json']:
        if path.exists():
            path.with_name(path.stem + '_before_parallel_' + stamp + '.json').write_bytes(path.read_bytes())
    with (args.run / 'parallel_preparation.log').open('a') as prep_log, (args.run / 'queue.log').open('a') as queue_log:
        preparation_command = [python, str(ROOT / 'tools/vera_libero/training_worker.py'),
                               '--data', str(args.data), '--gpus', args.gpus]
        if args.batches:
            preparation_command.extend(['--batches',args.batches])
        preparation = subprocess.Popen(preparation_command, cwd=ROOT, stdout=prep_log, stderr=subprocess.STDOUT)
        # Publish the live PID before queue startup; preparation replaces this
        # with its full operational record as soon as its interpreter starts.
        temporary = args.data / 'worker_status.json.tmp'
        temporary.write_text(json.dumps({'status': 'RUNNING', 'pid': preparation.pid,
                                         'detail': 'Starting user-authorized parallel workers',
                                         'training_started': False}) + '\n')
        temporary.replace(args.data / 'worker_status.json')
        queue_command = [python, str(ROOT / 'tools/vera_libero/train_when_ready.py'),
                         '--data', str(args.data), '--run', str(args.run)]
        queue = subprocess.Popen(queue_command, cwd=ROOT, stdout=queue_log, stderr=subprocess.STDOUT)
        record = {'pipeline_pid': os.getpid(), 'parent_pid': os.getppid(),
                  'preparation_pid': preparation.pid, 'queue_pid': queue.pid,
                  'gpus': args.gpus, 'batches':args.batches,'preparation_command': preparation_command,
                  'queue_command': queue_command, 'training_started': False,
                  'started_at': datetime.now().astimezone().isoformat()}
        (args.run / 'pipeline.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        preparation_code = preparation.wait()
        # The training queue observes readiness itself and ceases training
        # supervision through its bounded callback; this only reaps processes.
        queue_code = queue.wait()
        record.update(preparation_exit_code=preparation_code, queue_exit_code=queue_code,
                      finished_at=datetime.now().astimezone().isoformat())
        (args.run / 'pipeline.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
