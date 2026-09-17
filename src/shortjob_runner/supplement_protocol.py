"""Versioned, single-GPU supplemental matrices; independent of historical results."""
from __future__ import annotations

import hashlib
import itertools
import json

PROTOCOL = 'supplement-v1'
FAMILIES = (
    'fixed_shape_infer_small', 'prefill_toy', 'decode_toy', 'cv_online_infer',
    'llm_decode_proxy', 'dlrm_recommendation', 'rl_policy_infer', 'synthetic_kernel_chain',
)
MODELS = {
    'resnet50': 'ResNet50_Weights.IMAGENET1K_V2',
    'vit_b_16': 'ViT_B_16_Weights.IMAGENET1K_V1',
    'mobilenet_v3_large': 'MobileNet_V3_Large_Weights.IMAGENET1K_V2',
}


def task_id(task: dict) -> str:
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:20]


def validate_single_gpu(count: int) -> None:
    if count != 1:
        raise ValueError(f'Exactly one CUDA device must be visible; got {count}. Set CUDA_VISIBLE_DEVICES.')


def task(workload, action, batch, seed, steps, *, checkpoints=None, jobs=1,
         kind='trajectory', phase='correctness', repeat=0):
    return dict(protocol=PROTOCOL, workload=workload, action=action, batch=batch,
                seed=seed, steps=steps, checkpoints=checkpoints or [steps], jobs=jobs,
                kind=kind, phase=phase, repeat=repeat)


def core_tasks(*, controls=True) -> list[dict]:
    rows = []
    for w, a, b, s in itertools.product(FAMILIES, ['graphs_only', 'graphs_input_copy'], [1, 4, 16, 32], [17, 42]):
        rows.append(task(w, a, b, s, 500, checkpoints=[1, 10, 50, 100, 500]))
    for a, b, s in itertools.product(['graphs_only', 'graphs_input_copy'], [1, 16], [17, 42]):
        rows.append(task('short_train_small', a, b, s, 100, checkpoints=[1, 3, 10, 50, 100]))
    for w, a, n in itertools.product(['cv_online_infer', 'llm_decode_proxy'], ['graphs_only', 'graphs_input_copy'], [10, 500]):
        rows.append(task(w, a, 16, 42, n, kind='independent'))
    for w, a, n in itertools.product(['cv_online_infer', 'llm_decode_proxy'], ['graphs_only', 'graphs_input_copy'], [50, 500]):
        rows.append(task(w, a, 16, 42, n, jobs=8, kind='reuse'))
    if controls:
        rows = [task(w, a, 4, 42, 10, kind='baseline_control')
                for w, a in itertools.product(FAMILIES, ['eager', 'best_eager'])] + [
                    task('short_train_small', 'eager', 1, 42, 10, kind='baseline_control')
                ] + rows
        rows += [task(w, a, 4, 42, 3, kind='precheck_control') for w, a in itertools.product(
            ['dynamic_shape_infer_small', 'short_text_transformer_dynamic', 'gnn_irregular'],
            ['graphs_only', 'graphs_input_copy'])]
    return rows


def real_tasks(models, *, batch=1, repeats=5, lengths=(10, 50, 100, 500)) -> list[dict]:
    if batch < 1 or repeats < 1 or not lengths or any(n < 1 for n in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError('batch, repeats and lengths must be positive')
    if not models or len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError('Choose unique supported models')
    rows = []
    for m, a, n in itertools.product(models, ['eager', 'graphs_input_copy', 'compile_only'], lengths):
        for seed in [17, 42]:
            rows.append(task(m, a, batch, seed, n, kind='real_images', checkpoints=sorted({1, n}), phase='correctness'))
        for repeat in range(repeats):
            rows.append(task(m, a, batch, 42, n, kind='real_images', phase='performance', repeat=repeat))
    return rows
