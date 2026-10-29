"""Offline, versioned reinterpretation of existing measurements only."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import statistics

from shortjob_runner.policies import evaluate_policy_on_group, group_rows, policy_set
from shortjob_runner.summarize import (
    EAGER_FAMILY_ACTIONS, attach_oracle_and_regret, eager_baseline_row, group_key,
)
from shortjob_runner.queue_replay import (
    build_policy_jobs, eligible_conditions, index_action_rows, index_policy_rows,
    mean_or_none, percentile, replay_single_server, sample_trace,
)
from shortjob_runner.graph_reuse import condition_key, summarize_conditions, summarize_device

VARIANTS = ('legacy', 'fixed_input', 'matched_eager')


def prepare_summaries(rows, variant, blocked):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    groups = defaultdict(list)
    seen = set()
    for source in rows:
        key = (group_key(source), source['action'])
        if key in seen:
            raise ValueError(f'Duplicate condition/action: {key}')
        seen.add(key)
        total = source.get('median_total_runtime_s')
        if source['action_feasible'] and (total is None or not math.isfinite(total) or total <= 0):
            raise ValueError(f'Invalid feasible service time: {key}')
        groups[key[0]].append(dict(source))
    output = []
    for group, items in groups.items():
        if variant != 'legacy':
            items = [r for r in items if r['action'] != 'graphs_input_copy']
        baseline = eager_baseline_row(items, blocked_actions=blocked)
        if baseline is None:
            raise ValueError(f'Missing valid eager baseline: {group}')
        if variant == 'matched_eager':
            items = [r for r in items if r['action'] not in EAGER_FAMILY_ACTIONS]
            # Select the predetermined safe implementation, never the fastest
            # target timing. Both calibration buckets and oracle see this row.
            items.append({**baseline, 'action': 'eager',
                          'measured_source_action': baseline['action']})
        for item in items:
            item.setdefault('measured_source_action', item['action'])
        output.extend(items)
    return attach_oracle_and_regret(output, blocked_actions=blocked)


class CanonicalEagerPolicy:
    needs_action_rows = True

    def __init__(self, policy):
        self.policy = policy
        if hasattr(policy, 'information_budget'):
            self.information_budget = policy.information_budget

    def __call__(self, row, actions):
        choice = (self.policy(row, actions) if getattr(self.policy, 'needs_action_rows', False)
                  else self.policy(row))
        return 'eager' if choice in EAGER_FAMILY_ACTIONS else choice


def evaluate_variant(rows, variant, blocked, names=None):
    policies = policy_set(100, rows, blocked_actions=blocked)
    if variant != 'legacy':
        policies.pop('always_graphs_input_copy')
        policies.pop('always_cpu')
    if names is not None:
        policies = {name: policies[name] for name in names}
    if variant == 'matched_eager':
        policies = {name: CanonicalEagerPolicy(policy) for name, policy in policies.items()}
    output = []
    for group, actions in group_rows(rows).items():
        for name, policy in policies.items():
            result = evaluate_policy_on_group(group, actions, name, policy, blocked_actions=blocked)
            chosen = actions.get(result['chosen_action'])
            result['measured_source_action'] = chosen.get('measured_source_action') if chosen else None
            output.append(result)
    return output


def normalized_trace(rows, num_jobs, seed, target, workload_weights, step_weights):
    index = index_action_rows(rows)
    conditions = eligible_conditions(index)
    if not conditions or num_jobs < 1 or not math.isfinite(target) or target <= 0:
        raise ValueError('A trace needs conditions, jobs and positive target utilization')
    times, weights = [], []
    for key in conditions:
        ref = next(iter(index[key].values()))
        times.append(float(ref['oracle_total_runtime_s']))
        weights.append(float((workload_weights or {}).get(ref['workload_id'], 1)) *
                       float((step_weights or {}).get(ref['num_steps'], 1)))
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Weights must be finite, nonnegative, with positive sum')
    weighted_mean = sum(t*w for t,w in zip(times, weights, strict=True)) / sum(weights)
    # Reuse the original random sequence. Correct its uniform-mean arrival
    # normalization explicitly, without changing the legacy helper or artifacts.
    adjusted_target = target * statistics.fmean(times) / weighted_mean
    rate, trace = sample_trace(conditions, index, num_jobs, seed, adjusted_target,
                              workload_weights=workload_weights, num_steps_weights=step_weights)
    return rate, trace, weighted_mean


def queue_on_trace(rows, policy_rows, trace, rate, blocked, metadata):
    index = index_action_rows(rows)
    policies = index_policy_rows(policy_rows)
    if not trace or rate <= 0 or not math.isfinite(rate):
        raise ValueError('Invalid arrival trace or rate')
    if any(trace[i]['arrival_time_s'] < trace[i-1]['arrival_time_s'] for i in range(1, len(trace))):
        raise ValueError('Arrival times must be nondecreasing')
    trace_hash = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
    oracle_jobs = build_policy_jobs(trace, 'oracle_best_action', policies, index,
                                   blocked_actions=blocked)
    oracle_completion = replay_single_server(oracle_jobs)['per_job_completion_times']
    outputs = []
    for name in sorted(policies):
        jobs = build_policy_jobs(trace, name, policies, index, blocked_actions=blocked)
        if any(not math.isfinite(j['service_time_s']) or j['service_time_s'] <= 0 for j in jobs):
            raise ValueError(f'Invalid service times for {name}')
        replay = replay_single_server(jobs)
        completion = replay.pop('per_job_completion_times')
        gaps = [a-b for a,b in zip(completion, oracle_completion, strict=True)]
        service = round(statistics.fmean(j['service_time_s'] for j in jobs), 6)
        offered = round(rate * service, 6)
        failures = sum(bool(policies[name][j['condition']]['policy_failed']) for j in trace)
        outputs.append({
            **metadata, 'policy': name, 'trace_sha256': trace_hash, 'num_jobs': len(trace),
            'arrival_rate': rate, 'arrival_rate_lambda': rate, **replay,
            'effective_gpu_busy_time_s': None, 'mean_service_time_s': service,
            'effective_offered_load': offered, 'policy_unstable': offered >= 1,
            'stability_status': 'unstable_rho_ge_1' if offered >= 1 else 'stable_rho_lt_1',
            'total_optimization_overhead_s': round(sum(j['optimization_overhead_s'] for j in jobs), 6),
            'wasted_optimization_overhead_s': round(sum(j['wasted_optimization_overhead_s'] for j in jobs), 6),
            'wrong_admit_count': sum(bool(j['wrong_admit']) for j in jobs),
            'missed_opportunity_count': sum(bool(j['missed_opportunity']) for j in jobs),
            'policy_failure_count': failures,
            'fallback_count': sum(j['chosen_action'] != j['admitted_action'] for j in jobs),
            'failure_recovery_cost_s': 0.0,
            'oracle_gap_mean': mean_or_none(gaps), 'oracle_gap_p95': percentile(gaps, .95),
        })
    return outputs


def prepare_reuse(raw, variant, blocked, label):
    if variant not in VARIANTS:
        raise ValueError(variant)
    grouped = defaultdict(list)
    seen = set()
    for r in raw:
        key = (condition_key(r), r['action'], r['repeat'])
        if key in seen:
            raise ValueError(f'Duplicate reuse repeat: {key}')
        seen.add(key)
        grouped[key[0]].append({**r, 'device_label': label})
    selected = []
    for items in grouped.values():
        if variant != 'legacy':
            items = [r for r in items if r['action'] != 'graphs_input_copy']
        if variant == 'matched_eager':
            wid = items[0]['workload_id']
            baseline = next((a for a in ('best_eager', 'eager') if any(
                r['action'] == a and r['action_feasible'] and
                r.get('amortized_runtime_per_job_s') is not None and (wid,a) not in blocked
                for r in items)), None)
            if baseline is None:
                raise ValueError(f'Missing reuse baseline for {wid}')
            items = [{**r, 'measured_source_action': r['action'],
                      'action': 'eager' if r['action'] == baseline else r['action']}
                     for r in items if r['action'] not in EAGER_FAMILY_ACTIONS or r['action'] == baseline]
        selected.extend(items)
    conditions = summarize_conditions(selected, metric='amortized_runtime_per_job_s',
                                      blocked_actions_by_device={label: blocked})
    return conditions, summarize_device(label, conditions)
