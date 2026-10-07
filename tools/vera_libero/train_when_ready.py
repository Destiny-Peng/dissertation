"""Queue authorized training after preparation; initial observation is bounded."""
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def record(run, status, **extra):
    value = {'status': status, 'updated_at': datetime.now().astimezone().isoformat(),
             'queue_pid': os.getpid(), **extra}
    temporary = run / 'queue_status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(run / 'queue_status.json')
    report = ROOT / 'environment_reports' / ('LIBERO_JIDM_TRAINING_' + run.name + '.md')
    report.write_text('# LIBERO J-IDM training\n\n```json\n' + json.dumps(value, indent=2) + '\n```\n\n'
                      f'Run: {run.relative_to(ROOT)}. Initial training health and ETA will be written '
                      'to training_health.json and TRAINING_STATUS.md in the run directory.\n')
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observe-steps', type=int, default=200)
    args = parser.parse_args()
    assert args.data.resolve().is_relative_to(ROOT / 'datasets')
    assert args.run.resolve().is_relative_to(ROOT / 'outputs')
    args.run.mkdir(parents=True, exist_ok=True)
    record(args.run, 'WAITING_FOR_VERIFIED_DATA', training_started=False, data=str(args.data.relative_to(ROOT)))
    while True:
        verification = args.data / 'verification.json'
        status = json.loads((args.data / 'worker_status.json').read_text())
        if status['status'] == 'READY' and verification.exists() and json.loads(verification.read_text()).get('passed'):
            break
        if status['status'] == 'BLOCKED':
            record(args.run, 'BLOCKED_BY_PREPARATION', training_started=False, detail=status)
            return
        try:
            os.kill(status['pid'], 0)
        except ProcessLookupError:
            record(args.run, 'PREPARATION_PROCESS_EXITED', training_started=False, detail=status)
            return
        time.sleep(30)
    while shutil.disk_usage(ROOT).free < 35 * 1024 ** 3:
        record(args.run, 'WAITING_FOR_DISK', training_started=False, required_free_gib=35)
        time.sleep(30)
    while True:
        gpu_rows = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'
        ], text=True).strip().splitlines()
        free = [(int(row.split(',')[1]), int(row.split(',')[0])) for row in gpu_rows]
        available, gpu = max(free)
        if available >= 30 * 1024:
            break
        record(args.run, 'WAITING_FOR_GPU', training_started=False, free_gpu_mib=available, required_free_mib=30 * 1024)
        time.sleep(30)
    env = os.environ.copy()
    env['LIBERO_JIDM_GPU'] = str(gpu)
    env['PYTHONPATH'] = str(ROOT / 'tools') + os.pathsep + env.get('PYTHONPATH', '')
    command = [str(ROOT / 'tools/run_libero_jidm_train.sh'), str(args.data),
               'experiment.training.max_steps=10000',
               'experiment.training.batch_size=2',
               'experiment.training.optim.accumulate_grad_batches=8',
               'experiment.training.enable_progress_bar=false',
               f'experiment.validation.val_every_n_step={args.observe_steps * 8}',
               'experiment.training.checkpointing.every_n_train_steps=200',
               f'hydra.run.dir={args.run}',
               '+lf3r_observer._target_=vera_libero.training_observer.TrainingObserver',
               f'+lf3r_observer.run_dir={args.run}', f'+lf3r_observer.observe_steps={args.observe_steps}']
    with (args.run / 'training.log').open('w') as log:
        log.write('COMMAND ' + json.dumps(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record(args.run, 'TRAINING_LAUNCHED', training_started=True, training_pid=process.pid,
               gpu=gpu, free_gpu_mib=available, command=command, max_optimizer_steps=10000)
        # No polling of training: the in-process callback stops observation after
        # verified validation/checkpoint/progress. This wait only reaps the child.
        code = process.wait()
    record(args.run, 'TRAINING_FINISHED' if code == 0 else 'TRAINING_FAILED',
           training_started=True, exit_code=code, training_log=str((args.run / 'training.log').relative_to(ROOT)))


if __name__ == '__main__':
    main()
