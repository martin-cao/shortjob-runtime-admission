from collections import defaultdict
import statistics as st

from camera_ready_data import read_rows, require, stage_records, write_json

MODELS = ['mobilenet_v3_large', 'resnet50', 'vit_b_16']
HORIZONS = [10, 50, 100, 500]
ACTIONS = ['eager', 'graphs_input_copy', 'compile_only']
COMPONENTS = ['initialization_s', 'optimization_setup_s', 'execution_s']


def analyze_sessions(root, manifest, output):
    reports, points = {}, []
    measurements = defaultdict(list)
    for name, info in manifest['sessions'].items():
        records = stage_records(root, info['stages']['real'])
        groups = defaultdict(list)
        for row in records:
            if row['status'] == 'measured':
                task = row['task']
                groups[task['workload'], task['steps'], task['action']].append(row)
        require(len(groups) == 36, f'{name}: action 条件数量错误')
        for group in groups.values():
            group.sort(key=lambda r: r['task']['repeat'])
            require([r['task']['repeat'] for r in group] == list(range(5)), f'{name}: 正式重复不完整')
        conditions = []
        for model in MODELS:
            for steps in HORIZONS:
                samples = {a: [r['total_s'] for r in groups[model, steps, a]] for a in ACTIONS}
                e, g, c = [samples[a] for a in ACTIONS]
                em, gm, cm = map(st.median, [e, g, c])
                conditions.append({'model': model, 'steps': steps,
                    'graph_time_change_percent': 100 * (gm / em - 1), 'graph_speedup': em / gm,
                    'graph_saved_s': em - gm, 'graph_observed_ranges':
                        'all_graph_samples_faster' if max(g) < min(e) else
                        'all_eager_samples_faster' if min(g) > max(e) else 'overlap',
                    'graph_same_round_changes_percent': [100 * (gg / ee - 1) for ee, gg in zip(e, g, strict=True)],
                    'compile_slowdown': cm / em, 'compile_penalty_s': cm - em,
                    'all_compile_samples_slower': min(c) > max(e),
                    'actions': {a: {'median_total_s': st.median(samples[a]), 'min_total_s': min(samples[a]),
                        'max_total_s': max(samples[a]), 'sample_cv_percent': 100 * st.stdev(samples[a]) / st.mean(samples[a]),
                        'component_medians_s': {k: st.median(r[k] for r in groups[model, steps, a]) for k in COMPONENTS},
                        'samples': samples[a]} for a in ACTIONS}})
                point = {'model': model, 'session': name, 'median': 100 * (gm / em - 1),
                         'low': 100 * (min(g) / max(e) - 1), 'high': 100 * (max(g) / min(e) - 1)}
                if steps == 500:
                    points.append(point)
                device = '4060' if info['device'] == 'laptop' else info['device']
                measurements[device, model, steps].append({**point, 'eager_median_s': em, 'graph_median_s': gm,
                    'compile_ratio': cm / em, 'compile_slower_in_all_samples': min(c) > max(e), 'samples_per_action': 5})
        cvs = [100 * st.stdev(r['total_s'] for r in group) / st.mean(r['total_s'] for r in group) for group in groups.values()]
        reports[name] = {'performance_rows': 180, 'correctness_rows': 72, 'conditions': conditions,
            'total_cv_percent': {'median_across_conditions': st.median(cvs), 'max': max(cvs),
                                 'conditions_above_5': sum(v > 5 for v in cvs), 'conditions_above_10': sum(v > 10 for v in cvs)}}
        write_json(output / f'{name}.json', reports[name])
    aggregate = []
    for (device, model, steps), blocks in measurements.items():
        aggregate.append({'device': device, 'model': model, 'steps': steps,
            'median': st.median(b['median'] for b in blocks), 'low': min(b['low'] for b in blocks),
            'high': max(b['high'] for b in blocks), 'blocks': blocks,
            'samples_per_action': sum(b['samples_per_action'] for b in blocks),
            'compile_ratio': st.median(b['compile_ratio'] for b in blocks),
            'compile_slower_in_all_samples': all(b['compile_slower_in_all_samples'] for b in blocks)})
    require(len(aggregate) == 48 and len(points) == 27, '真实模型绘图范围错误')
    print('九个真实模型会话：648 条正确性记录、1620 次正式计时已重建', flush=True)
    return reports, {'fig_real_sessions': points, 'real_model_results': aggregate}


def heatmap(controlled_root):
    workloads = ['fixed_shape_infer_small', 'cv_online_infer', 'llm_decode_proxy', 'decode_toy',
                 'prefill_toy', 'dlrm_recommendation', 'rl_policy_infer', 'synthetic_kernel_chain']
    excluded = {'short_train_small': 'numerical failure', 'short_text_transformer_padded': 'capture failure',
                'short_text_transformer_dynamic': 'precheck rejection', 'dynamic_shape_infer_small': 'precheck rejection',
                'gnn_irregular': 'precheck rejection'}
    devices = ['4060', 'v100', 'a100', 'h100']
    counts = [[0] * 16 for _ in workloads]
    for j, device in enumerate(devices):
        rows = read_rows(controlled_root / device / 'cold/action_summary.jsonl')
        require({r['workload_id'] for r in rows} == set(workloads) | set(excluded), 'workload 覆盖不一致')
        require(not any(r['oracle_action'] == 'graphs_only' for r in rows if r['workload_id'] in excluded),
                '资格排除集合出现 graph oracle')
        for i, workload in enumerate(workloads):
            oracle = {(r['batch_size'], r['num_steps']): r['oracle_action'] for r in rows if r['workload_id'] == workload}
            require(len(oracle) == 16, 'workload 条件数量不完整')
            for k, steps in enumerate(HORIZONS):
                counts[i][j * 4 + k] = sum(action == 'graphs_only' for (_, n), action in oracle.items() if n == steps)
    return {'workloads': workloads, 'counts': counts, 'excluded': excluded,
            'columns': [[device, n] for device in devices for n in HORIZONS]}
