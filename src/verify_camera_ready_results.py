import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from camera_ready_controlled import MIXES
from camera_ready_data import PACKAGES, read_json, read_rows, require, write_json
from legacy_reanalysis import normalized_trace
from shortjob_runner.summarize import group_key


def close(actual, expected, tolerance=1e-6):
    require(abs(actual - expected) <= tolerance, f'独立计算不一致: {actual} != {expected}')


def key(row):
    return row['workload_id'], row['batch_size'], row['num_steps']


def verify(data_root, output):
    manifest = read_json(data_root / 'manifest.json')
    report = read_json(output / 'analysis.json')
    checks = defaultdict(int)
    base = data_root / manifest['legacy']['root'] / 'raw'
    panels = [(base / package, output / 'controlled' / device, report['controlled'][device],
               ['cold_core', 'warm_reuse_core', 'correctness_gate']) for device, package in PACKAGES.items()]
    panels.append((data_root / 'replication/v100_c', output / 'replication', report['replication'],
                   ['cold', 'warm', 'legacy_gate']))
    for source, folder, panel, names in panels:
        blocked = {(r['workload_id'], r['action']) for r in read_rows(source / f'{names[2]}.jsonl')
                   if r.get('correctness_failed')}
        for suite, filename in zip(['cold', 'warm'], names[:2], strict=True):
            values = defaultdict(list)
            for row in read_rows(source / f'{filename}.jsonl'):
                if row['status'] == 'completed' and row['action_feasible']:
                    values[key(row), row['action']].append(row['total_runtime_s'])
            medians = {k: round(float(np.median(v)), 6) for k, v in values.items()}
            summaries = read_rows(folder / suite / 'action_summary.jsonl')
            oracles = {}
            for condition in {key(r) for r in summaries}:
                candidates = {action: value for (k, action), value in medians.items() if k == condition
                              and action not in {'eager', 'best_eager', 'graphs_input_copy'}
                              and (condition[0], action) not in blocked}
                baseline = 'eager' if condition[0] == 'short_train_small' else 'best_eager'
                candidates['eager'] = medians[condition, baseline]
                oracles[condition] = min(candidates, key=candidates.get)
            for summary in summaries:
                condition = key(summary)
                if summary['median_total_runtime_s'] is not None:
                    close(summary['median_total_runtime_s'], medians[condition, summary['measured_source_action']], 1e-12)
                    checks['raw_medians'] += 1
                require(summary['oracle_action'] == oracles[condition], '独立 oracle 不一致')
            policies = read_rows(folder / suite / 'policy_eval.jsonl')
            by_policy = defaultdict(list)
            indexed = {(key(r), r['action']): r for r in summaries}
            for row in policies:
                by_policy[row['policy']].append(row)
                if row['policy_failed'] or row.get('policy_correctness_failure'):
                    continue
                condition = key(row)
                best = indexed[condition, oracles[condition]]['median_total_runtime_s']
                chosen = indexed[condition, row['chosen_action']]['median_total_runtime_s']
                close(row['normalized_regret_vs_oracle'], max(0., (chosen - best) / best))
                checks['valid_policy_regrets'] += 1
            for name, rows in by_policy.items():
                valid = [r for r in rows if not r['policy_failed'] and not r.get('policy_correctness_failure')]
                if valid:
                    p90 = float(np.quantile([r['normalized_regret_vs_oracle'] for r in valid], .9, method='lower'))
                    close(p90, panel[suite]['policy_metrics'][name]['p90_normalized_regret'], 1e-12)
            if suite == 'cold':
                cold = summaries
                choices = {(group_key(r), r['policy']): r for r in policies}
                service = {(group_key(r), r['action']): r['median_total_runtime_s'] for r in summaries}
        traces = {}
        for queue in read_rows(folder / 'queue_replay.jsonl'):
            trace_key = queue['job_mix_id'], queue['target_utilization'], queue['queue_seed']
            if trace_key not in traces:
                ww, sw = MIXES[trace_key[0]]
                traces[trace_key] = normalized_trace(cold, 1000, trace_key[2], trace_key[1], ww, sw)[1]
            finish, waits, completions, services = 0., [], [], []
            for job in traces[trace_key]:
                choice = choices[job['condition'], queue['policy']]
                action = 'eager' if choice['policy_failed'] or choice.get('policy_correctness_failure') else choice['chosen_action']
                duration = service[job['condition'], action]
                arrival = job['arrival_time_s']
                wait = max(finish - arrival, 0.)
                finish = arrival + wait + duration
                waits.append(wait)
                completions.append(wait + duration)
                services.append(duration)
            close(queue['mean_completion_time_s'], float(np.mean(completions)))
            close(queue['p95_completion_time_s'], float(np.quantile(completions, .95, method='lower')))
            close(queue['mean_wait_time_s'], float(np.mean(waits)))
            close(queue['worker_busy_time_s'], float(np.sum(services)))
            close(queue['throughput_jobs_per_s'], 1000 / finish)
            checks['queue_policy_traces'] += 1
    for name, info in manifest['sessions'].items():
        rows_path = data_root / info['stages']['real']['path'] / 'rows'
        values = defaultdict(list)
        for path in rows_path.glob('*.json'):
            row = read_json(path)
            if row['task']['phase'] == 'performance':
                task = row['task']
                close(row['total_s'], sum(row[k] for k in ['initialization_s', 'optimization_setup_s', 'execution_s']), 1e-10)
                values[task['workload'], task['steps'], task['action']].append(row['total_s'])
        for condition in report['real_sessions'][name]['conditions']:
            for action, summary in condition['actions'].items():
                samples = np.array(values[condition['model'], condition['steps'], action])
                require(len(samples) == 5, '真实模型重复数量错误')
                for metric, value in [('median_total_s', np.median(samples)), ('min_total_s', np.min(samples)),
                                      ('max_total_s', np.max(samples)), ('sample_cv_percent', 100 * np.std(samples, ddof=1) / np.mean(samples))]:
                    close(summary[metric], float(value), 1e-10)
                checks['real_action_conditions'] += 1
    result = {'status': 'verified', 'checks': dict(checks),
              'queue_scope': 'Independent FCFS recurrence and service selection; shared frozen arrival generator.'}
    write_json(output / 'independent_verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description='直接从正式测量独立核验中位数、regret、queue 和真实模型统计')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root.resolve(), args.results.resolve()), indent=2))


if __name__ == '__main__':
    main()
