import argparse
import csv
import json
from pathlib import Path
import statistics

from camera_ready_controlled import analyze_panel
from camera_ready_data import LABELS, PACKAGES, read_json, read_rows, require, sha, verify_dataset, write_json
from camera_ready_real import analyze_sessions, heatmap


def compare(actual, expected, location):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f'{location}: 类型不一致')
        for key, value in expected.items():
            require(key in actual, f'{location}: 缺少字段 {key}')
            compare(actual[key], value, f'{location}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{location}: 列表长度不一致')
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare(a, e, f'{location}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(abs(actual - expected) <= 1e-11 * max(1, abs(expected)), f'{location}: {actual} != {expected}')
    else:
        require(actual == expected, f'{location}: {actual!r} != {expected!r}')


def validate_reference(root, reports, replication, sessions, plots):
    # 预期值仅在全部测量重新计算完成后读入。
    expected = read_json(root / 'reference/expected.json')
    checked = []
    for device, report in reports.items():
        for name, value in [('policy_metrics', report['cold']['policy_metrics']), ('rank', report['rank']),
                            ('information_budget', report['information_budget']),
                            ('prediction_diagnostic', report['prediction_diagnostic'])]:
            compare(value, expected[name][device], f'{device}.{name}')
            checked.append(f'{device}.{name}')
        for suite, key in [('cold', 'cold_core'), ('warm', 'warm_reuse_core')]:
            compare(report[suite]['oracle_counts'], expected['matched_oracles'][device][key], f'{device}.{suite}.oracle')
        for policy, devices in expected['queue_statistics'].items():
            compare(report['queue_statistics'][policy], devices[device], f'{device}.queue.{policy}')
    reference = expected['replication']['controlled']['october']
    for suite in ['cold', 'warm']:
        for field in ['conditions', 'oracle_counts', 'policy_metrics', 'statuses']:
            compare(replication[suite][field], reference[suite][field], f'replication.{suite}.{field}')
    compare(replication['reuse'], reference['reuse'], 'replication.reuse')
    compare(replication['queue'], reference['queue'], 'replication.queue')
    for name, report in sessions.items():
        wanted = expected['real_sessions'][name]
        compare(report['total_cv_percent'], wanted['total_cv_percent'], f'{name}.cv')
        by_key = {(r['model'], r['steps']): r for r in report['conditions']}
        for condition in wanted['conditions']:
            scientific = {k: v for k, v in condition.items() if k != 'actions'}
            compare(by_key[condition['model'], condition['steps']], scientific, f'{name}.{condition["model"]}.{condition["steps"]}')
    for field in ['fig_oracle_action_mix', 'fig_real_sessions', 'real_model_results']:
        compare(plots[field], expected['plots'][field], field)
    checked.extend(['replication.cold', 'replication.warm', 'replication.reuse', 'replication.queue',
                    'nine_real_sessions', 'figure_heatmap', 'figure_session_points', 'real_model_block_aggregation'])
    return {'passed': True, 'checks': checked, 'paper_source_sha256': expected['paper_source_sha256']}


def export_tables(output, reports, replication, sessions):
    folder = output / 'tables'
    folder.mkdir()
    policy_rows = [{'configuration': device, 'policy': policy, **row}
                   for device, report in {**reports, 'v100_replication': replication}.items()
                   for policy, row in report['cold']['policy_metrics'].items()]
    queue_rows = [{'configuration': device, **row}
                  for device, report in {**reports, 'v100_replication': replication}.items() for row in report['queue']]
    real_rows = [{'session': session, **{k: v for k, v in row.items()
                                      if k not in ['actions', 'graph_same_round_changes_percent']}}
                 for session, report in sessions.items() for row in report['conditions']]
    boundaries = [{'configuration': device, **report['rank'],
                   'reuse_graph_wins': report['reuse']['graph_oracle_count'],
                   'reuse_gain': report['reuse']['median_graphs_only_speedup_vs_eager'],
                   'loo_oracle_changes': report['repeat_sensitivity']['conditions_changing_oracle_in_any_omission']}
                  for device, report in {**reports, 'v100_replication': replication}.items()]
    improvements = []
    for device, report in {**reports, 'v100_replication': replication}.items():
        heavy = {(r['job_mix_id'], r['policy']): r for r in report['queue'] if r['load_level'] == 'heavy'}
        for mix in ['uniform', 'graph_friendly_heavy', 'dynamic_irregular_heavy', 'short_step_heavy']:
            eager = heavy[mix, 'always_eager']['p95_completion_time_s']
            admitted = heavy[mix, 'static_gate_plus_eligibility']['p95_completion_time_s']
            improvements.append({'configuration': device, 'mix': mix, 'eager_p95_s': eager,
                                 'eligibility_p95_s': admitted, 'reduction_percent': 100 * (1 - admitted / eager)})
    for name, rows in [('policies.csv', policy_rows), ('queues.csv', queue_rows), ('real_models.csv', real_rows),
                       ('boundaries.csv', boundaries), ('queue_improvements.csv', improvements)]:
        with (folder / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def compare_panels(old_paths, new_paths, output):
    comparisons = {}
    for suite in ['cold', 'warm', 'reuse', 'gate']:
        fields = ['workload_id', 'action', 'batch_size', 'num_steps'] + ([] if suite == 'gate' else ['repeat'])
        old = {tuple(r[k] for k in fields): r for r in read_rows(old_paths[suite])}
        new = {tuple(r[k] for k in fields): r for r in read_rows(new_paths[suite])}
        require(old.keys() == new.keys(), '完整复现的任务集合与原始会话不一致')
        changes = [{'key': list(k), 'old_status': old[k]['status'], 'new_status': new[k]['status']}
                   for k in old if old[k]['status'] != new[k]['status'] or old[k]['action_feasible'] != new[k]['action_feasible']]
        comparisons[suite] = {'matched_records': len(old), 'status_changes': changes}
    condition = lambda r: (r['workload_id'], r['batch_size'], r['num_steps'])
    for suite in ['cold', 'warm']:
        before = read_rows(output / 'controlled/v100' / suite / 'action_summary.jsonl')
        after = read_rows(output / 'replication' / suite / 'action_summary.jsonl')
        ia = {(condition(r), r['action']): r for r in before}
        ib = {(condition(r), r['action']): r for r in after}
        ga, gb = {condition(r): r for r in before}, {condition(r): r for r in after}
        drifts = [{'action': action, 'change_percent': 100 * (r['median_total_runtime_s'] / ia[k, action]['median_total_runtime_s'] - 1)}
                  for (k, action), r in ib.items() if r['action_feasible'] and ia[k, action]['action_feasible']]
        flips = [{'condition': list(k), 'old_oracle': ga[k]['oracle_action'], 'new_oracle': r['oracle_action'],
                  'old_graph_ratio': ia[k, 'graphs_only']['median_total_runtime_s'] / ia[k, 'eager']['median_total_runtime_s'],
                  'new_graph_ratio': ib[k, 'graphs_only']['median_total_runtime_s'] / ib[k, 'eager']['median_total_runtime_s']}
                 for k, r in gb.items() if r['oracle_action'] != ga[k]['oracle_action']]
        comparisons[suite].update(oracle_flips=flips, median_drift_by_action={
            action: statistics.median(r['change_percent'] for r in drifts if r['action'] == action)
            for action in sorted({r['action'] for r in drifts})})
    write_json(output / 'replication_comparison.json', comparisons)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description='从累计正式数据重建 camera-ready 实验结果')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='尚不存在的输出目录')
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.out.resolve()
    require(not output.exists(), '输出目录必须尚不存在')
    require(not output.is_relative_to(root), '输出目录不能位于输入数据中')
    quality = verify_dataset(root)
    manifest = read_json(root / 'manifest.json')
    output.mkdir(parents=True)
    write_json(output / 'data_validation.json', quality)
    base = root / manifest['legacy']['root'] / 'raw'
    reports, panel_paths = {}, {}
    for device, package in PACKAGES.items():
        paths = {name: base / package / filename for name, filename in
                 [('gate', 'correctness_gate.jsonl'), ('cold', 'cold_core.jsonl'),
                  ('warm', 'warm_reuse_core.jsonl'), ('reuse', 'graph_reuse_core.jsonl')]}
        reports[device] = analyze_panel(paths, output / 'controlled' / device, LABELS[device])
        panel_paths[device] = paths
    paths = {name: root / manifest['replication'][field] for name, field in
             [('gate', 'legacy_gate'), ('cold', 'cold'), ('warm', 'warm'), ('reuse', 'graph_reuse')]}
    replication = analyze_panel(paths, output / 'replication', 'V100', replication=True)
    comparison = compare_panels(panel_paths['v100'], paths, output)
    sessions, plots = analyze_sessions(root, manifest, output / 'real')
    plots['fig_oracle_action_mix'] = heatmap(output / 'controlled')
    write_json(output / 'plotted_values.json', plots)
    result = {'controlled': reports, 'replication': replication, 'real_sessions': sessions}
    write_json(output / 'analysis.json', result)
    export_tables(output, reports, replication, sessions)
    validation = validate_reference(root, reports, replication, sessions, plots)
    compare(comparison, read_json(root / 'reference/expected.json')['replication']['controlled_comparison'], 'replication_comparison')
    validation['checks'].append('replication_status_and_oracle_comparison')
    verify_dataset(root)
    validation.update({'dataset_manifest_sha256': sha(root / 'manifest.json'), 'gpu_experiments_run': False,
                       'cold_conditions': 832, 'real_sessions': 9, 'real_performance_rows': 1620,
                       'replication_records': 9583})
    write_json(output / 'validation.json', validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
