"""GT achieved-motion -> OSC diagnostic; never used to produce IDM predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, read_plan, write_json, metric_to_osc, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = read_plan(args.output)
    release = ROOT / plan['checkpoint'] / 'config.yaml'
    import yaml
    dataset = yaml.safe_load(release.read_text())['dataset']
    pred, gt = [], []
    for task in plan['tasks']:
        for demo in task['demos']:
            target = args.output / demo['output']
            ref = np.load(target / 'reference.npz')
            metric = ref['gt_metric']
            normalized_grip = metric[:, 6] * 80. / dataset['action_abs_scale'][6]
            raw, action = metric_to_osc(metric, ref['ee_pos'][:-1], ref['gripper'][:-1],
                                        normalized_grip, task['controller'])
            pred.append(action); gt.append(ref['gt_action'])
            write_json(target / 'gt_motion_to_osc_diagnostic.json', {'action': metrics(action, ref['gt_action']),
                'arm_action': metrics(action[:, :6], ref['gt_action'][:, :6]),
                'note': 'Oracle achieved next motion, not a prediction; conversion/controller-lag diagnostic only'})
    pred, gt = np.concatenate(pred), np.concatenate(gt)
    data = {'action': metrics(pred, gt), 'arm_action': metrics(pred[:, :6], gt[:, :6]),
            'gripper_accuracy': float((pred[:, 6] == gt[:, 6]).mean()),
            'note': 'Diagnostic uses future GT proprio deliberately; never used in learned predictions or execution.'}
    write_json(args.output / 'gt_motion_to_osc_diagnostic.json', data)
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    main()
