#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../project_env.sh"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export MPLCONFIGDIR="${CACHE}/matplotlib"
LIBERO_PREP_DATA="$(cat "${CACHE}/libero_jidm_training_current_run.txt")"
LIBERO_TRAIN_RUN="$(cat "${CACHE}/libero_jidm_training_authorized_run.txt")"
python3 -S - "${SCRIPT_DIR}/vera_libero/parallel_pipeline.py" "${LIBERO_PREP_DATA}" "${LIBERO_TRAIN_RUN}" "${LIBERO_JIDM_PREP_GPUS:-0,1,2}" "${LIBERO_JIDM_PREP_BATCHES:-2,4,1}" <<'PY'
import os, sys
from pathlib import Path
script, data, run, gpus, batches = sys.argv[1:]
run = Path(run)
pid_path = run / 'pipeline.pid'
if pid_path.exists():
    pid = int(pid_path.read_text())
    command = Path(f'/proc/{pid}/cmdline')
    if command.exists() and b'parallel_pipeline.py' in command.read_bytes():
        raise SystemExit(f'Pipeline already running: PID {pid}')
child = os.fork()
if child:
    os.waitpid(child, 0)
    print(f'Detached pipeline launched; PID record: {pid_path}')
    raise SystemExit(0)
os.setsid()
if os.fork():
    os._exit(0)
input_fd = os.open('/dev/null', os.O_RDONLY)
output_fd = os.open(str(run / 'pipeline.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(input_fd, 0)
os.dup2(output_fd, 1)
os.dup2(output_fd, 2)
os.close(input_fd)
os.close(output_fd)
os.execv(sys.executable, [sys.executable, '-S', script, '--data', data, '--run', str(run), '--gpus', gpus, '--batches', batches])
PY
