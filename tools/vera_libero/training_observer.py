"""Observe the start of an authorized run, then disable monitoring in-place."""
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
from lightning.pytorch import Callback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class TrainingObserver(Callback):
    def __init__(self, run_dir, observe_steps=200):
        self.run_dir = Path(run_dir)
        self.observe_steps = int(observe_steps)
        self.active = True
        self.steps = []
        self.losses = []
        self.gradients = []
        self.validation = None
        self.validation_started = None
        self.validation_seconds = 0.0
        self.parameter = None
        self.parameter_before = None
        self.parameter_indices = None
        self.parameter_changed = False

    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()
        write_json(self.run_dir / 'training_health.json', {
            'status': 'OBSERVING', 'training_started': True,
            'max_steps': trainer.max_steps, 'observe_steps': self.observe_steps,
            'started_at': datetime.now().astimezone().isoformat(),
        })

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.active:
            return
        parameters = [p for p in pl_module.parameters() if p.grad is not None]
        if not parameters:
            raise RuntimeError('No gradients in LIBERO J-IDM training')
        norms = torch.stack([
            torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters
        ])
        norm = torch.linalg.vector_norm(norms).item()
        if not math.isfinite(norm):
            raise FloatingPointError('Non-finite training gradient norm')
        self.gradients.append(norm)
        if self.parameter is None:
            self.parameter = parameters[int(norms.argmax())]
            gradients = self.parameter.grad.detach().reshape(-1).abs()
            self.parameter_indices = gradients.topk(min(4096, gradients.numel())).indices
            self.parameter_before = self.parameter.detach().reshape(-1)[self.parameter_indices].float().cpu().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.active:
            return
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        if loss is not None:
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite training loss')
            self.losses.append(value)
        step = int(trainer.global_step)
        if not self.steps or step != self.steps[-1][0]:
            self.steps.append((step, time.monotonic()))
            if self.parameter is not None:
                current = self.parameter.detach().reshape(-1)[self.parameter_indices].float().cpu()
                self.parameter_changed |= not torch.equal(current, self.parameter_before)
            if step and (step <= 5 or step % 20 == 0):
                self._record(trainer)
        self._maybe_finish(trainer)

    def on_validation_start(self, trainer, pl_module):
        if self.active and not trainer.sanity_checking:
            self.validation_started = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        if not self.active or trainer.sanity_checking:
            return
        metrics = {str(k): float(v.detach()) for k, v in trainer.callback_metrics.items()
                   if str(k).startswith('loss/validation') and torch.is_tensor(v) and v.numel() == 1}
        if not metrics or not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Missing or non-finite held-out validation losses')
        self.validation = metrics
        if self.validation_started is not None:
            self.validation_seconds += time.monotonic() - self.validation_started
        self._maybe_finish(trainer)

    def _record(self, trainer, status='OBSERVING'):
        state = {
            'status': status, 'updated_at': datetime.now().astimezone().isoformat(),
            'training_started': True, 'optimizer_steps': int(trainer.global_step),
            'microbatches_observed': len(self.losses), 'max_steps': trainer.max_steps,
            'recent_loss_mean': sum(self.losses[-20:]) / max(1, len(self.losses[-20:])),
            'initial_loss_mean': sum(self.losses[:20]) / max(1, len(self.losses[:20])),
            'gradient_norm_min': min(self.gradients, default=0),
            'gradient_norm_max': max(self.gradients, default=0),
            'parameter_update_verified': self.parameter_changed,
            'validation_losses': self.validation,
            'finite_losses_and_gradients': True,
            'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'cuda_reserved_gib': torch.cuda.memory_reserved() / 1024**3,
            'elapsed_seconds': time.monotonic() - self.started,
        }
        if status == 'HEALTHY_MONITORING_STOPPED':
            points = [(s, t) for s, t in self.steps if s >= 5]
            elapsed = time.monotonic() - points[0][1]
            seconds_per_step = elapsed / (int(trainer.global_step) - points[0][0])
            remaining = seconds_per_step * (trainer.max_steps - int(trainer.global_step))
            state.update(seconds_per_optimizer_step=seconds_per_step,
                         estimated_remaining_hours=remaining / 3600,
                         estimated_finish_at=(datetime.now().astimezone() + timedelta(seconds=remaining)).isoformat(),
                         eta_note='Estimate includes observed validation/checkpoint overhead; shared GPU/I/O load may change.',
                         monitoring_stopped=True)
        write_json(self.run_dir / 'training_health.json', state)
        return state

    def _maybe_finish(self, trainer):
        if int(trainer.global_step) < self.observe_steps or self.validation is None:
            return
        if not self.parameter_changed or not any(g > 0 for g in self.gradients):
            raise RuntimeError('Training progressed without a verified parameter update')
        checkpoints = list((self.run_dir / 'checkpoints').glob('*.ckpt'))
        if not checkpoints or not all(p.stat().st_size > 0 for p in checkpoints):
            return
        state = self._record(trainer, 'HEALTHY_MONITORING_STOPPED')
        state['checkpoint_files'] = [str(p) for p in checkpoints]
        write_json(self.run_dir / 'training_health.json', state)
        report = self.run_dir / 'TRAINING_STATUS.md'
        report.write_text(
            '# LIBERO J-IDM training\n\nTraining is progressing normally; initial monitoring has stopped.\n\n'
            f'Optimizer steps: {state["optimizer_steps"]}/{state["max_steps"]}. '
            f'Finite loss/gradients, parameter update, held-out validation and checkpoint writing verified.\n\n'
            f'Estimated remaining time: {state["estimated_remaining_hours"]:.1f} hours; '
            f'estimated finish: {state["estimated_finish_at"]}. Shared GPU and I/O load may change this estimate.\n\n'
            'The trainer continues independently. Initial health does not establish action accuracy or simulator success.\n'
        )
        root = Path(__file__).resolve().parents[2]
        environment_report = root / 'environment_reports' / ('LIBERO_JIDM_TRAINING_' + self.run_dir.name + '.md')
        environment_report.write_text(report.read_text() + '\nRun: ' + str(self.run_dir.relative_to(root)) + '\n')
        queue_path = self.run_dir / 'queue_status.json'
        if queue_path.exists():
            queue = json.loads(queue_path.read_text())
            queue.update(status='HEALTHY_MONITORING_STOPPED', estimated_remaining_hours=state['estimated_remaining_hours'],
                         estimated_finish_at=state['estimated_finish_at'], monitoring_stopped=True)
            write_json(queue_path, queue)
        with (root / 'SETUP_STATUS.md').open('a') as stream:
            stream.write(f'\n| {state["updated_at"]} | LIBERO J-IDM | authorized_finetuning | 1 | RUNNING / INITIAL MONITORING STOPPED | '
                         f'{state["optimizer_steps"]} optimizer steps; finite loss/gradient, parameter updates, validation and checkpoint passed. '
                         f'ETA {state["estimated_remaining_hours"]:.1f} hours. Report {environment_report.relative_to(root)}. |\n')
        print('LF3R_INITIAL_MONITORING_COMPLETE ' + json.dumps(state), flush=True)
        self.active = False
        self.parameter = self.parameter_before = None
        self.parameter_indices = None
        self.steps.clear()
        self.losses.clear()
        self.gradients.clear()

    def on_exception(self, trainer, pl_module, exception):
        if self.active:
            write_json(self.run_dir / 'training_health.json', {
                'status': 'FAILED_DURING_INITIAL_OBSERVATION',
                'updated_at': datetime.now().astimezone().isoformat(),
                'optimizer_steps': int(trainer.global_step),
                'exception': repr(exception), 'monitoring_stopped': False,
            })
