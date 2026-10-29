from collections import Counter, defaultdict
import statistics as st

import numpy as np

from camera_ready_data import read_rows, require, write_json, write_rows
from legacy_reanalysis import evaluate_variant, normalized_trace, prepare_reuse, prepare_summaries, queue_on_trace
from paper_metrics import summarize_policies
from shortjob_runner.policies import AmortizationPolicy, ACTION_ORDER, ACTION_RANK, GRAPH_ACTIONS
from shortjob_runner.summarize import group_key, load_correctness_blocklist, summarize_raw
from summarize_rank_stability import summarize_conditions, summarize_device

MIXES = {
    'uniform': (None, None),
    'graph_friendly_heavy': ({'llm_decode_proxy': 5, 'decode_toy': 4,
                              'fixed_shape_infer_small': 3, 'synthetic_kernel_chain': 3}, None),
    'dynamic_irregular_heavy': ({'dynamic_shape_infer_small': 5, 'gnn_irregular': 5,
                                 'short_text_transformer_dynamic': 4}, None),
    'short_step_heavy': (None, {10: 6, 50: 3, 100: 1, 500: .5}),
}
LOADS = {'light': .35, 'medium': .65, 'heavy': .9}
REPLICATION_POLICIES = ['always_eager', 'always_compile', 'always_graphs', 'job_length_only',
                        'shape_only', 'static_threshold', 'static_gate_only',
                        'static_gate_plus_eligibility', 'amortization_policy', 'oracle_best_action']
SENSITIVITY_POLICIES = ['always_eager', 'always_compile', 'static_threshold', 'static_gate_only',
                        'static_gate_plus_eligibility', 'amortization_policy', 'oracle_best_action']
QUEUE_FIELDS = ['mean_completion_time_s', 'p95_completion_time_s', 'p99_completion_time_s',
                'mean_wait_time_s', 'p95_wait_time_s', 'throughput_jobs_per_s',
                'worker_busy_time_s', 'effective_offered_load', 'total_optimization_overhead_s',
                'wasted_optimization_overhead_s', 'wrong_admit_count', 'missed_opportunity_count',
                'oracle_gap_mean', 'oracle_gap_p95', 'arrival_rate', 'policy_failure_count', 'fallback_count']


def queue_aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arrival_basis'], row['job_mix_id'], row['load_level'], row['policy']].append(row)
    return [dict(zip(['arrival_basis', 'job_mix_id', 'load_level', 'policy'], key),
                 seeds=len(group), unstable_traces=sum(r['policy_unstable'] for r in group),
                 **{f: round(st.fmean(r[f] for r in group), 6) for f in QUEUE_FIELDS})
            for key, group in sorted(groups.items())]


def policy_comparison(rows):
    indexed = defaultdict(dict)
    for row in rows:
        indexed[row['policy']][group_key(row)] = row['admitted_action']
    calibrated = indexed['amortization_policy']
    eligible = indexed['static_gate_plus_eligibility']
    require(calibrated.keys() == eligible.keys(), '策略条件集合不一致')
    return {'conditions': len(calibrated), 'action_differences': sum(calibrated[k] != eligible[k] for k in calibrated)}


def prediction_diagnostic(rows, blocked):
    groups = defaultdict(dict)
    for row in rows:
        groups[group_key(row)][row['action']] = row
    reports = []
    for key, actions in groups.items():
        row = actions['eager']
        n = row['num_steps']
        training = [r for r in rows if group_key(r) != key]
        model = AmortizationPolicy(training, blocked_actions=blocked)
        stats = model.stats[model._stats_key(row), 'eager']
        before = stats['startup_cost_s'] + n * stats['step_time_s'] + stats['risk_penalty_s']
        retained = [r for r in training if r['action'] == 'eager' and model._stats_key(r) == model._stats_key(row)]
        require(len(retained) == 3, '预测诊断需要三个非目标 horizon')
        x = np.array([[1, r['num_steps']] for r in retained])
        y = np.array([r['median_total_runtime_s'] for r in retained])
        intercept, slope = np.linalg.lstsq(x, y, rcond=None)[0]
        accepted = bool(intercept >= 0 and slope >= 0)
        after = float(intercept + n * slope) if accepted else before
        scores = []
        for action in ACTION_ORDER:
            if (row['workload_id'], action) in blocked:
                continue
            bucket = model.stats.get((model._stats_key(row), action)) or model.stats.get((model._fallback_key(row), action))
            if not bucket or not bucket['action_feasible']:
                continue
            if action in GRAPH_ACTIONS and row['shape_stability'] == 'unstable':
                continue
            score = after if action == 'eager' else bucket['startup_cost_s'] + n * bucket['step_time_s'] + bucket['risk_penalty_s']
            scores.append((score, action))
        choice = min(scores, key=lambda pair: (pair[0], ACTION_RANK.get(pair[1], 100)))[1]
        original = model(row)
        reports.append({'workload': row['workload_id'], 'batch': row['batch_size'], 'steps': n,
                        'eager_actual': row['median_total_runtime_s'], 'eager_pred_original': before,
                        'eager_pred_intercept_diagnostic': after, 'training_N': [r['num_steps'] for r in retained],
                        'intercept': float(intercept), 'slope': float(slope), 'nonnegative_fit_used': accepted,
                        'original_choice': original, 'diagnostic_choice': choice,
                        'original_regret': actions[original]['normalized_regret_vs_oracle'],
                        'diagnostic_regret': actions[choice]['normalized_regret_vs_oracle']})
    return {'conditions': reports,
            'by_N': {str(n): {
                'median_eager_prediction_ratio_original': st.median(r['eager_pred_original'] / r['eager_actual'] for r in reports if r['steps'] == n),
                'median_eager_prediction_ratio_diagnostic': st.median(r['eager_pred_intercept_diagnostic'] / r['eager_actual'] for r in reports if r['steps'] == n),
                'choices_changed': sum(r['original_choice'] != r['diagnostic_choice'] for r in reports if r['steps'] == n)}
                     for n in [10, 50, 100, 500]},
            'fit_used': sum(r['nonnegative_fit_used'] for r in reports),
            'choices_changed': sum(r['original_choice'] != r['diagnostic_choice'] for r in reports),
            'mean_regret_original': st.mean(r['original_regret'] for r in reports),
            'mean_regret_diagnostic': st.mean(r['diagnostic_regret'] for r in reports)}


def repeat_sensitivity(raw, full, blocked):
    original = {group_key(row): row['oracle_action'] for row in full}
    policies = evaluate_variant(full, 'matched_eager', blocked, names=SENSITIVITY_POLICIES)
    choices = {(group_key(r), r['policy']): r['admitted_action'] for r in policies}
    traces = [normalized_trace(full, 1000, seed, .9, None, None) for seed in range(1, 6)]
    changed, reports = set(), []
    for omit in range(1, 6):
        subset = prepare_summaries(summarize_raw([r for r in raw if r['repeat'] != omit], blocked), 'matched_eager', blocked)
        selected = evaluate_variant(subset, 'matched_eager', blocked, names=SENSITIVITY_POLICIES)
        oracle = {group_key(row): row['oracle_action'] for row in subset}
        changed.update(k for k in oracle if oracle[k] != original[k])
        queues = []
        for seed, (rate, trace, _) in enumerate(traces, 1):
            queues.extend(queue_on_trace(subset, selected, trace, rate, blocked,
                {'arrival_basis': 'full_matched_medians_fixed_arrivals', 'job_mix_id': 'uniform',
                 'load_level': 'heavy', 'queue_seed': seed, 'target_utilization': .9}))
        reports.append({'omitted_repeat': omit, 'oracle_counts': dict(Counter(oracle.values())),
                        'oracle_changes': sum(oracle[k] != original[k] for k in oracle),
                        'policy_choice_changes': dict(Counter(r['policy'] for r in selected
                            if r['admitted_action'] != choices[group_key(r), r['policy']])),
                        'policy_metrics': summarize_policies(selected), 'queue_aggregate': queue_aggregate(queues)})
    return {'conditions_changing_oracle_in_any_omission': len(changed), 'omissions': reports}


def analyze_panel(paths, output, label, replication=False):
    blocked = load_correctness_blocklist(paths['gate'])
    result, summaries, policies, raw_suites = {}, {}, {}, {}
    names = REPLICATION_POLICIES if replication else None
    for suite, count in [('cold', 208), ('warm', 24)]:
        raw = read_rows(paths[suite])
        base = summarize_raw(raw, blocked)
        rows = prepare_summaries(base, 'matched_eager', blocked)
        selected = evaluate_variant(rows, 'matched_eager', blocked, names=names)
        groups = {group_key(r): r for r in rows}
        require(len(groups) == count, f'{label}/{suite}: 条件集合不完整')
        write_rows(output / suite / 'action_summary.jsonl', rows)
        write_rows(output / suite / 'policy_eval.jsonl', selected)
        result[suite] = {'conditions': len(groups), 'oracle_counts': dict(Counter(r['oracle_action'] for r in groups.values())),
                         'policy_metrics': summarize_policies(selected), 'calibration_comparison': policy_comparison(selected),
                         'statuses': dict(Counter(r['status'] for r in raw))}
        summaries[suite], policies[suite], raw_suites[suite] = rows, selected, raw
    cold_original = {(r['workload_id'], r['batch_size'], r['num_steps'], r['action']): r
                     for r in summarize_raw(raw_suites['cold'], blocked)}
    warm_original = summarize_raw(raw_suites['warm'], blocked)
    result['warm_cold_ratios'] = {action: st.median(r['median_total_runtime_s'] / cold_original[
        r['workload_id'], r['batch_size'], r['num_steps'], action]['median_total_runtime_s']
        for r in warm_original if r['action'] == action and r['action_feasible'])
        for action in ['compile_only', 'compile_reduce_overhead']}
    reuse, reuse_summary = prepare_reuse(read_rows(paths['reuse']), 'matched_eager', blocked, label)
    write_json(output / 'reuse_conditions.json', reuse)
    result['reuse'] = reuse_summary
    rank = summarize_conditions(summaries['cold'], device_label=label, marginal_threshold=.05, blocked_actions=blocked)
    result['rank'] = summarize_device(label, rank)
    write_json(output / 'rank_conditions.json', rank)
    eager = [r for r in summaries['cold'] if r['action'] == 'eager']
    prefix = [r for r in eager if r['median_first_k_eager_step_time_ms'] is not None]
    result['information_budget'] = {'prefix_rows': len(prefix), 'full_horizon_average_rows': len(eager) - len(prefix),
                                    'prefix_workloads': sorted({r['workload_id'] for r in prefix}),
                                    'prefix_k_values': sorted({r['median_first_k_steps'] for r in prefix})}
    queues = []
    basis = 'session_specific_matched_oracle' if replication else 'revised_mix_specific_oracle_reference'
    for mix, (ww, sw) in MIXES.items():
        for load, target in LOADS.items():
            for seed in range(1, 6):
                rate, trace, mean = normalized_trace(summaries['cold'], 1000, seed, target, ww, sw)
                queues.extend(queue_on_trace(summaries['cold'], policies['cold'], trace, rate, blocked,
                    {'arrival_basis': basis, 'job_mix_id': mix, 'load_level': load, 'queue_seed': seed,
                     'target_utilization': target, 'normalization_oracle_mean_s': mean}))
    write_rows(output / 'queue_replay.jsonl', queues)
    result['queue'] = queue_aggregate(queues)
    result['queue_statistics'] = {}
    for policy in sorted({r['policy'] for r in queues}):
        group = [r for r in queues if r['policy'] == policy and r['job_mix_id'] == 'uniform' and r['load_level'] == 'heavy']
        result['queue_statistics'][policy] = {'mean': st.mean(r['p95_completion_time_s'] for r in group),
            'sample_sd': st.stdev(r['p95_completion_time_s'] for r in group),
            'mean_load': st.mean(r['effective_offered_load'] for r in group)}
    result['repeat_sensitivity'] = repeat_sensitivity(raw_suites['cold'], summaries['cold'], blocked)
    if not replication:
        result['prediction_diagnostic'] = prediction_diagnostic(summaries['cold'], blocked)
    write_json(output / 'analysis.json', result)
    print(f'{label}: controlled、warm、reuse、queue 与重复敏感性已重建', flush=True)
    return result
