"""Create a 200-demo view over cached packs, verify, then launch authorized training."""
import argparse
import collections
import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.data / 'split_manifest.json').read_text())
    selected = set(json.loads((args.data / 'selection_200.json').read_text())['episode_ids'])
    episodes = [dict(ep) for ep in original['episodes'] if ep['episode_id'] in selected]
    assert len(episodes) == 200
    subset = args.data.with_name(args.data.name + '_subset200')
    subset.mkdir(exist_ok=True)
    for ep in episodes:
        packed = args.data / ep['path']
        assert packed.exists() and packed.with_suffix('.json').exists(), ep['episode_id']
        ep['path'] = os.path.relpath(packed, subset)
    counts = dict(collections.Counter(ep['split'] for ep in episodes))
    pairs = {split: sum(ep['pairs'] for ep in episodes if ep['split'] == split)
             for split in counts}
    save(subset / 'split_manifest.json', {
        **original, 'selection': 'Fixed 200-demo allowlist; original trajectory split labels preserved',
        'counts': counts, 'pairs': pairs, 'episodes': episodes,
        'task_counts': dict(collections.Counter(ep['source_hdf5'] for ep in episodes)),
        'limitation': 'Uneven task coverage; some tasks have no held-out validation demos.',
    })
    cfg = OmegaConf.load(args.data / 'configs/libero_idm.yaml')
    for split, key in [('training', 'dataset'), ('validation', 'validation_dataset'), ('test', 'test_dataset')]:
        save(subset / split / 'index.json', [os.path.relpath(subset / ep['path'], subset / split)
                                           for ep in episodes if ep['split'] == split])
        root = '${oc.env:PROJECT_ROOT}/' + str((subset / split).relative_to(ROOT))
        cfg[key].root = root
        cfg[key].data_root = root
    # Dataset length is finite; max_steps alone must control the requested run.
    cfg.experiment.training.max_epochs = -1
    (subset / 'configs').mkdir(exist_ok=True)
    OmegaConf.save(cfg, subset / 'configs/libero_idm.yaml')
    save(subset / 'preparation.json', {'status': 'VERIFYING', 'counts': counts, 'pairs': pairs})
    save(subset / 'worker_status.json', {'status': 'VERIFYING', 'pid': os.getpid(), 'training_started': False})
    print('SUBSET_READY_FOR_VERIFICATION', counts, pairs, flush=True)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    free, gpu = max((int(row.split(',')[1]), int(row.split(',')[0])) for row in rows)
    if free < 16 * 1024:
        raise RuntimeError('Insufficient free GPU memory for verification')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    try:
        with (args.run / 'subset_verification.log').open('w') as log:
            subprocess.run([os.environ['LF3R_VERA_JIDM_PYTHON'],
                            str(ROOT / 'tools/vera_libero/verify_training.py'),
                            '--data', str(subset)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        save(subset / 'worker_status.json', {'status': 'READY', 'pid': os.getpid(), 'training_started': False})
        subprocess.run([os.environ['LF3R_VERA_JIDM_PYTHON'],
                        str(ROOT / 'tools/vera_libero/train_when_ready.py'),
                        '--data', str(subset), '--run', str(args.run)], cwd=ROOT, check=True)
    except BaseException as error:
        save(subset / 'worker_status.json', {'status': 'BLOCKED', 'pid': os.getpid(), 'detail': str(error)})
        raise


if __name__ == '__main__':
    main()
