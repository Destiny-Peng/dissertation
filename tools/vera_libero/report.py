"""Aggregate complete transfer runs; preserve task/demo weighting and paired outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT, DIMENSIONS, read_plan, write_json, metrics


def wilson(success, total):
    p, z = success / total, 1.96
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [float(center-half), float(center+half)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = read_plan(args.output)
    rows, predicted, gt, predicted_metric, gt_metric = [], [], [], [], []
    one_pred, one_gt = [], []
    by_task = []
    for task in plan['tasks']:
        task_rows = []
        for demo in task['demos']:
            target = args.output / demo['output']
            p = np.load(target / 'prediction.npz')
            prediction = json.loads((target / 'prediction_metrics.json').read_text())
            playback = json.loads((target / 'playback_metrics.json').read_text())
            alignment = json.loads((target / 'alignment.json').read_text())
            assert alignment['passed']
            assert len(p['predicted_action']) == demo['pairs']
            assert np.array_equal(p['action_index'], np.arange(1, demo['frames']))
            predicted.append(p['predicted_action']); gt.append(p['gt_action'])
            predicted_metric.append(p['predicted_metric_delta']); gt_metric.append(p['gt_metric_delta'])
            single = np.load(target / 'one_step_playback.npz')
            one_pred.append(single['predicted_errors']); one_gt.append(single['gt_errors'])
            row = {'task': task['task_name'], 'demo': demo['demo'], 'pairs': demo['pairs'],
                   'action_mae': prediction['action']['mae'], 'action_mse': prediction['action']['mse'],
                   'arm_action_mae': prediction['arm_action']['mae'], 'gripper_accuracy': prediction['gripper_accuracy'],
                   'clipped_fraction': prediction['clipped_fraction'],
                   'gt_final_success': playback['gt']['final_success'], 'idm_final_success': playback['idm']['final_success'],
                   'gt_ever_success': playback['gt']['ever_success'], 'idm_ever_success': playback['idm']['ever_success'],
                   'gt_position_error_m': playback['gt']['mean_position_error_m'],
                   'idm_position_error_m': playback['idm']['mean_position_error_m'],
                   'one_step_gt_position_m': playback['one_step']['gt_mean_position_error_m'],
                   'one_step_idm_position_m': playback['one_step']['pred_mean_position_error_m'],
                   'paired_actual_position_mean_m': playback['paired_actual_motion']['mean_position_error_m']}
            rows.append(row); task_rows.append(row)
        by_task.append({'task': task['task_name'], 'demos': len(task_rows),
                        'pairs': sum(r['pairs'] for r in task_rows),
                        'gt_successes': sum(r['gt_final_success'] for r in task_rows),
                        'idm_successes': sum(r['idm_final_success'] for r in task_rows),
                        'demo_macro_action_mae': float(np.mean([r['action_mae'] for r in task_rows]))})
    pred, truth = np.concatenate(predicted), np.concatenate(gt)
    pm, tm = np.concatenate(predicted_metric), np.concatenate(gt_metric)
    one_p, one_g = np.concatenate(one_pred), np.concatenate(one_gt)
    n = len(rows); gs = sum(r['gt_final_success'] for r in rows); ps = sum(r['idm_final_success'] for r in rows)
    import yaml
    dataset = yaml.safe_load((ROOT / plan['checkpoint'] / 'config.yaml').read_text())['dataset']
    physical_scale = np.asarray(dataset['action_abs_scale']) / np.asarray([50.] * 6 + [80.])
    normalized_state_pred, normalized_state_gt = pm / physical_scale, tm / physical_scale
    interface_diagnostic = json.loads((args.output / 'gt_motion_to_osc_diagnostic.json').read_text())
    results = {'gt_motion_to_osc_diagnostic': interface_diagnostic, 'demos': n, 'tasks': len(by_task), 'pairs': len(pred),
               'action_micro': metrics(pred, truth), 'arm_action_micro': metrics(pred[:, :6], truth[:, :6]),
               'zero_action_micro': metrics(np.zeros_like(truth), truth),
               'zero_arm_action_micro': metrics(np.zeros_like(truth[:, :6]), truth[:, :6]),
               'state_delta_micro': metrics(pm, tm),
               'normalized_state_delta_micro': metrics(normalized_state_pred, normalized_state_gt),
               'zero_normalized_state_delta_micro': metrics(np.zeros_like(normalized_state_gt), normalized_state_gt), 'zero_state_delta_micro': metrics(np.zeros_like(tm), tm),
               'action_demo_macro_mae': float(np.mean([r['action_mae'] for r in rows])),
               'gripper_accuracy': float((pred[:, 6] == truth[:, 6]).mean()),
               'gt_final_successes': gs, 'idm_final_successes': ps,
               'gt_final_success_rate': gs/n, 'idm_final_success_rate': ps/n,
               'gt_success_wilson_95': wilson(gs,n), 'idm_success_wilson_95': wilson(ps,n),
               'paired_gt_success_idm_failure': sum(r['gt_final_success'] and not r['idm_final_success'] for r in rows),
               'paired_gt_failure_idm_success': sum(not r['gt_final_success'] and r['idm_final_success'] for r in rows),
               'one_step_pred_mean': one_p.mean(0).tolist(), 'one_step_gt_mean': one_g.mean(0).tolist(),
               'free_playback_pred_position_mean_m': float(np.mean([r['idm_position_error_m'] for r in rows])),
               'free_playback_gt_position_mean_m': float(np.mean([r['gt_position_error_m'] for r in rows])),
               'paired_actual_position_mean_m': float(np.mean([r['paired_actual_position_mean_m'] for r in rows])),
               'task_metrics': by_task,
               'scope': '50 pre-registered full demos; adjacent recorded RGB; paired free open-loop simulator playback; no finetuning or WM',
               'limitations': ['camera names match but extrinsics/scenes differ; no calibration or geometry adaptation',
                               'state delta is achieved motion, not the OSC command; fixed unit conversion cannot invert controller dynamics',
                               'gripper hold commands are visually ambiguous; fixed sign/deadband plus current qpos used',
                               'free playback actions are inferred on recorded demo images, not recomputed on deviating simulator images',
                               'per-camera independent serving mode was evaluated; joint multiview mode and task-specialized adaptive policy were not evaluated',
                               'Wilson intervals are nominal demo-level intervals; demonstrations share ten tasks and are not an independent random task sample',
                               'no WM-generated input or closed-loop corrective action evaluation']}
    write_json(args.output / 'summary.json', results)
    with (args.output / 'trajectories.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(7)
    axes[0].bar(x-.18, results['action_micro']['per_dimension_mae'], width=.36, label='MimicGen J-IDM')
    axes[0].bar(x+.18, results['zero_action_micro']['per_dimension_mae'], width=.36, label='Zero action')
    axes[0].set_xticks(x, DIMENSIONS); axes[0].set_ylabel('LIBERO normalized action MAE');axes[0].legend()
    x = np.arange(len(by_task))
    axes[1].bar(x-.18, [r['gt_successes']/r['demos'] for r in by_task], width=.36, label='GT replay')
    axes[1].bar(x+.18, [r['idm_successes']/r['demos'] for r in by_task], width=.36, label='IDM replay')
    axes[1].set_xticks(x, [str(i) for i in x]);axes[1].set_xlabel('Task index in plan.json (sorted filenames)')
    axes[1].set_ylim(0,1.05);axes[1].set_ylabel('Final simulator success');axes[1].legend()
    fig.savefig(args.output / 'transfer.png', dpi=160);plt.close(fig)
    if ps < gs and results['arm_action_micro']['mae'] >= results['zero_arm_action_micro']['mae']:
        verdict = '未支持直接 zero-shot transfer：预测臂动作未优于零动作基线，且连续执行成功率低于 GT replay。'
    elif ps < gs * .8:
        verdict = '当前证据不支持直接用于 LIBERO corrective actions：连续执行成功率明显低于 GT replay。'
    else:
        verdict = '应根据以下运动误差与配对成功率判断；本测试不能单独证明 WM suffix corrective actions 可靠。'
    text = f'''# VERA MimicGen Panda J-IDM → LIBERO 评估

{verdict}

- 官方成功 demonstration：{n} 条，10 个任务，每任务5条；全部 {len(pred)} 个相邻 observation pair。
- GT continuous replay 最终成功：{gs}/{n} ({gs/n:.1%})；IDM continuous replay：{ps}/{n} ({ps/n:.1%})。
- LIBERO action overall MAE：{results['action_micro']['mae']:.6f}；MSE：{results['action_micro']['mse']:.6f}。
- Arm-only MAE：{results['arm_action_micro']['mae']:.6f}；零动作 arm-only MAE：{results['zero_arm_action_micro']['mae']:.6f}。
- Gripper accuracy：{results['gripper_accuracy']:.1%}。
- Native checkpoint 正规化状态增量 MAE：{results['normalized_state_delta_micro']['mae']:.6f}；零增量 MAE：{results['zero_normalized_state_delta_micro']['mae']:.6f}（独立于 OSC action 转换）。
- 逐状态恢复 single-step 位置误差：IDM {one_p[:,0].mean()*1000:.2f} mm；GT {one_g[:,0].mean()*1000:.2f} mm。
- 连续回放平均位置偏差（demo 宏平均）：IDM {results['free_playback_pred_position_mean_m']*1000:.2f} mm；GT {results['free_playback_gt_position_mean_m']*1000:.2f} mm。

## 对齐与模型

官方 HDF5 中 obs[k] 是执行 action[k] 后的图像和 proprio；因此 pair(k,k+1) 对应 action[k+1]，state 恢复为 states[k+1]。第一条 action 没有保存的前置 RGB，明确不计入 pair 预测。每条 demonstration 的 recorded XML 只修改资产文件路径，恢复对应相机/物体/动力学；采样的图像 PSNR 与 proprio 对齐检查均通过。GT replay 与 IDM replay 均从 states[1] 开始，运行相同的 T-1 步，不注入未来 simulator state。

checkpoint 为官方 idm-mimicgen-285ouq1q/model.ckpt，SHA256 和零 missing/unexpected 的严格加载见 model_load.json。输入为原始 OpenGL RGB、128x128、[0,1]，相机顺序为 agentview / wrist。与官方 serving 一致，每个相机独立计算 Jacobian。CoTracker3 offline 以真实两帧输入及15x15网格提取可见点运动；不使用更多未来帧或 GT action。正规化严格使用 checkpoint action_abs_scale/oflow_abs_scale；SE3×50 与左手指增量×80 按官方 DatasetConfig。官方 CoTracker 批量两帧路径的非连续张量 view 已改为等价 reshape，补丁保存于 tools/patches/cotracker3-batched-offline.patch。Tikhonov 正规化域求逆 lam=0.01，所有设置在计划/代码中预先固定，无 LIBERO 训练或拟合。

## Action 转换的含义

独立接口诊断：即使输入 GT 已实现状态增量，转换后的 arm-only action MAE 仍为 {interface_diagnostic['arm_action']['mae']:.6f}。这表明已实现运动不等于产生它的 OSC target command；该诊断刻意使用未来GT proprio，不参与IDM预测或执行。

VERA 预测的是 [translation(T1 inv(T0)), rotvec(R1 R0^-1), delta(left finger)]。先转换为物理增量，再通过当前末端位置还原 Cartesian dx；平移除以 OSC 的0.05 m、旋转除以0.5 rad，保留六个自由度，执行前限制到[-1,1]。Gripper 采用增量符号、固定0.18正规化死区与当前手指状态；未使用 GT gripper action。VERA task-specialized stacking controller 的 roll/pitch suppression、hand-tuned gains 和 online adaptive controller 未套用到 LIBERO，因此这是 frozen J-IDM 的通用单位转换评估，非整个 MimicGen policy 原封不动复制。

## 文件

- summary.json：micro / demo macro / per-task 指标、配对成功及95% Wilson区间。
- trajectories.csv：每条 demonstration 的预测误差和回放结果。
- trajectories/*/prediction.npz：预测action、未clip action、GT、signed/absolute/squared error、原始正规化预测、物理状态增量与GT、轨迹可见性/条件数。
- trajectories/*/actions.csv：逐pair/action index与七维pred/GT/error。
- trajectories/*/alignment.json、one_step_playback.npz、gt_playback.npz、idm_playback.npz、playback_metrics.json：simulator证据。
- 每个任务第一条demo的gt.mp4/idm.mp4：连续执行视频，两相机并排，显示帧率4 FPS（每5步取一帧，simulation 20Hz）。
- transfer.png：维度误差与任务成功率图。

## 结论边界

相同 Panda 不代表相机、场景或 OSC 反馈动态相同；状态增量与控制命令不是同一个量。纯视觉 pair 无法唯一恢复静止/接触时的 gripper hold command。连续执行使用真实demo上的teacher-forced预测动作；偏离后不会重新对当前sim图像推理，所以这项测试检验累计执行误差，不是closed-loop policy。它也不包含 WM生成图像。结果针对官方 per-camera serving 路径与此处固定接口，不排除 joint multiview、相机适配或 controller adaptation 能改善效果。95% Wilson 区间按demo计，仅作描述；50条demo共享10个任务，不是独立随机抽样的新任务。不能由此直接声称对 WM suffix 已可靠；如果当前真实帧都执行不可靠，下一步应先修正动作接口/相机适配或训练LIBERO IDM，而不是直接用于 corrective actions。
'''
    (args.output / 'REPORT.md').write_text(text)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
