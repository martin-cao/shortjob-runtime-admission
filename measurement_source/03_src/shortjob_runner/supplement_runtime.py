"""Matched-input and exact-N checking. Does not modify the historical runner."""
from __future__ import annotations

from contextlib import nullcontext
import time

import torch

from .actions import action_eligibility
from .correctness import make_seeded_workload
from .timing import synchronize


def snapshot(workload) -> dict[str, torch.Tensor]:
    state = dict(workload.correctness_state())
    model = getattr(workload, 'model', None)
    if model is not None:
        for name, param in model.named_parameters():
            name = name.removeprefix('_orig_mod.')
            state[f'param:{name}'] = param
            if workload.spec.supports_training:
                if param.grad is None:
                    raise RuntimeError(f'Missing training gradient: {name}')
                state[f'grad:{name}'] = param.grad
        for name, buffer in model.named_buffers():
            state[f'buffer:{name.removeprefix("_orig_mod.")}'] = buffer
    # Clone is required: detach alone still aliases graph output/parameter storage.
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def compare(reference, candidate, *, rtol=1e-4, atol=1e-5) -> dict:
    errors = {}
    for key in sorted(set(reference) | set(candidate)):
        if key not in reference or key not in candidate:
            errors[key] = {'reason': 'missing_key'}
            continue
        ref, got = reference[key], candidate[key]
        if ref.shape != got.shape or ref.dtype != got.dtype:
            errors[key] = {'reason': 'shape_or_dtype'}
            continue
        if not bool(torch.isfinite(ref).all() and torch.isfinite(got).all()):
            errors[key] = {'reason': 'nonfinite', 'max_abs': None, 'max_rel': None}
            continue
        delta = (ref.to(torch.float64) - got.to(torch.float64)).abs()
        maximum = float(delta.max()) if delta.numel() else 0.
        relative = float((delta / ref.to(torch.float64).abs().clamp_min(atol)).max()) if delta.numel() else 0.
        try:
            torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
            passed = True
        except AssertionError:
            passed = False
        errors[key] = {'passed': passed, 'max_abs': maximum, 'max_rel': relative}
    return {'passed': bool(errors) and all(v.get('passed', False) for v in errors.values()), 'errors': errors}


class Session:
    def __init__(self, workload, action, device):
        self.workload, self.action, self.device = workload, action, device
        self.graph = None
        if action not in {'eager', 'best_eager', 'graphs_only', 'graphs_input_copy', 'compile_only'}:
            raise ValueError(f'Unsupported supplemental action: {action}')

    def step(self):
        # Every inference action uses the same inference-mode contract.
        if self.workload.spec.supports_training or torch.is_inference_mode_enabled():
            self.workload.step()
        else:
            with torch.inference_mode():
                self.workload.step()

    def prepare(self):
        allowed, reason = action_eligibility(self.workload, self.action, self.device)
        if not allowed:
            raise ValueError(f'precheck: {reason}')
        if self.action == 'graphs_input_copy':
            allowed, reason = self.workload.input_copy_feasible()
            if not allowed:
                raise ValueError(f'precheck: {reason}')
        if self.action == 'compile_only':
            self.workload.enable_compile()
        if not self.action.startswith('graphs_'):
            return
        if self.device.type != 'cuda':
            raise ValueError('precheck: CUDA Graph requires CUDA')
        model = getattr(self.workload, 'model', None)
        restore = model is not None and self.workload.spec.supports_training
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()} if restore else {}
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            self.step()
        # Setup work is paid, but its state updates are not counted as job work.
        # Copy into existing storage; replacing tensors would invalidate capture.
        if restore:
            model.load_state_dict(initial)
        synchronize(self.device)
        self.graph = graph

    def advance(self, refresh):
        if refresh:
            self.workload.refresh_request_inputs()
        if self.graph is None:
            self.step()
        else:
            self.graph.replay()


def check_trajectory(workload_id, action, batch, seed, checkpoints, device, *,
                     refresh=None, jobs=1, factory=None) -> dict:
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[0] < 1 or jobs < 1:
        raise ValueError('Checkpoints must be unique, increasing positive steps; jobs must be positive')
    factory = factory or make_seeded_workload
    reference = factory(workload_id, batch, device, seed)
    candidate = factory(workload_id, batch, device, seed)
    for workload in [reference, candidate]:
        workload.seed_refresh(seed, device)
    refresh = (action == 'graphs_input_copy') if refresh is None else refresh
    ref = Session(reference, 'eager', device)
    cand = Session(candidate, action, device)
    cand.prepare()
    checks = []
    for job in range(1, jobs + 1):
        for step in range(1, checkpoints[-1] + 1):
            ref.advance(refresh)
            cand.advance(refresh)
            if step in checkpoints and (jobs == 1 or job in {1, 2, jobs}):
                synchronize(device)
                result = compare(snapshot(reference), snapshot(candidate))
                checks.append({'job': job, 'step': step, 'effective_steps': (job - 1) * checkpoints[-1] + step,
                               'comparison': result})
                if not result['passed']:
                    return {'status': 'failed', 'failure_stage': 'numerical', 'checks': checks,
                            'first_failing_checkpoint': {'job': job, 'step': step}}
    return {'status': 'passed', 'checks': checks,
            'training_setup_state_restored': cand.graph is not None and candidate.spec.supports_training,
            'input_contract': 'refreshed' if refresh else 'fixed', 'first_failing_checkpoint': None}


def measure(workload_id, action, batch, seed, steps, device, factory) -> dict:
    synchronize(device)
    start = time.perf_counter()
    workload = factory(workload_id, batch, device, seed)
    workload.seed_refresh(seed, device)
    synchronize(device)
    initialized = time.perf_counter()
    session = Session(workload, action, device)
    session.prepare()
    synchronize(device)
    prepared = time.perf_counter()
    with nullcontext() if workload.spec.supports_training else torch.inference_mode():
        for _ in range(steps):
            session.advance(refresh=True)
    synchronize(device)
    finished = time.perf_counter()
    return dict(status='measured', initialization_s=initialized-start,
                optimization_setup_s=prepared-initialized, execution_s=finished-prepared,
                total_s=finished-start, effective_steps=steps,
                input_contract='refreshed', timing_scope='worker_job_after_cuda_context_before_state_checks',
                final_state_finite=all(bool(torch.isfinite(v).all()) for v in snapshot(workload).values()))
