import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    data = args.data_root.resolve()
    results = args.results.resolve()
    report = read(results / 'analysis.json')
    plots = read(results / 'plotted_values.json')
    manifest = read(data / 'manifest.json')
    checks = []

    def check(location, actual, expected):
        if actual != expected:
            raise ValueError(f'{location}: {actual!r} != {expected!r}')
        checks.append(location)

    # 预期显示数值逐项取自最终提交 PDF；计算值来自重新读取的测量记录。
    devices = ['4060', 'v100', 'a100', 'h100']
    paths = list((data / manifest['legacy']['root'] / 'raw').glob('*/correctness_gate.jsonl'))
    gate_expected = {
        'eager': (13, 0, 0), 'best_eager': (12, 1, 0), 'graphs_only': (8, 3, 2),
        'graphs_input_copy': (8, 3, 2), 'compile_only': (13, 0, 0),
        'compile_reduce_overhead': (13, 0, 0), 'compile_plus_graphs': (9, 3, 1),
        'compile_reduce_overhead_plus_graphs': (0, 3, 10),
    }
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for action, expected in gate_expected.items():
            statuses = Counter(row['status'] for row in rows if row['action'] == action)
            check(f'Table I/{path.parent.name}/{action}',
                  tuple(statuses[name] for name in ['passed', 'infeasible', 'failed']), expected)
    for session, info in manifest['sessions'].items():
        for stage, description in info['stages'].items():
            if stage in ['core', 'supplement_core']:
                check(f'Table II/{session}/core', description['records'], 175)
                check(f'Table II/{session}/core-statuses', description['statuses'], {'passed': 169, 'infeasible': 6})
                tasks = read(data / description['path'] / 'manifest.json')['tasks']
                counts = Counter(task['kind'] if task['kind'] != 'trajectory' else
                                 'training' if task['workload'] == 'short_train_small' else 'inference'
                                 for task in tasks)
                check(f'Table II/{session}/coverage', dict(counts),
                      {'baseline_control': 17, 'inference': 128, 'training': 8,
                       'independent': 8, 'reuse': 8, 'precheck_control': 6})
            if stage == 'real':
                check(f'Table II/{session}/real', description['statuses'], {'passed': 72, 'measured': 180})

    boundaries = [(66, 37, 35, 10, '2.02', 16), (60, 22, 15, 11, '2.01', 7),
                  (81, 26, 52, 11, '3.02', 16), (50, 19, 25, 12, '1.12', 20)]
    for device, expected in zip(devices, boundaries, strict=True):
        panel = report['controlled'][device]
        rank = panel['rank']
        actual = (rank['graph_oracle_count'], rank['graph_oracle_marginal_count'],
                  rank['graph_oracle_ci95_overlap_count'], panel['reuse']['graph_oracle_count'],
                  f"{panel['reuse']['median_graphs_only_speedup_vs_eager']:.2f}",
                  panel['repeat_sensitivity']['conditions_changing_oracle_in_any_omission'])
        check(f'Table III/{device}', actual, expected)
        check(f'Cold policy equivalence/{device}', panel['cold']['calibration_comparison']['action_differences'], 0)
        check(f'Warm compile wins/{device}', sum(n for action, n in panel['warm']['oracle_counts'].items()
                                                if action.startswith('compile')), 0)

    heatmaps = [
        [[0,1,3,4],[0,0,0,2],[0,4,4,4],[1,3,4,4],[0,2,3,4],[0,3,3,4],[0,0,4,4],[1,0,0,4]],
        [[0,2,2,3],[0,0,0,2],[0,3,4,4],[0,4,4,4],[0,1,4,4],[0,3,3,4],[0,0,1,4],[0,0,0,4]],
        [[0,2,4,4],[2,1,1,2],[0,4,4,4],[0,4,4,4],[2,4,4,4],[0,4,4,4],[0,1,2,4],[0,0,4,4]],
        [[0,1,1,4],[0,0,1,4],[0,0,2,3],[0,1,3,4],[0,2,4,4],[0,1,3,3],[1,0,0,4],[0,0,0,4]],
    ]
    for i, device in enumerate(devices):
        check(f'Figure 2/{device}', [row[i * 4:i * 4 + 4] for row in plots['fig_oracle_action_mix']['counts']], heatmaps[i])
    for device, reduction in zip(devices, ['18.37', '30.89', '25.93', '18.45'], strict=True):
        point = next(p for p in plots['real_model_results'] if p['device'] == device
                     and p['model'] == 'mobilenet_v3_large' and p['steps'] == 500)
        check(f'Figure 3/MobileNet/{device}', f"{-point['median']:.2f}", reduction)
        check(f'Figure 3/MobileNet-envelope/{device}', point['high'] < 0, True)
    check('Real session count', len(report['real_sessions']), 9)
    check('Real timing count', sum(r['performance_rows'] for r in report['real_sessions'].values()), 1620)
    check('Real correctness count', sum(r['correctness_rows'] for r in report['real_sessions'].values()), 648)

    policies = {
        'always_eager': (208, 3, ['0.089','0.142','0.323','0.112'], ['0.0']*4, ['31.7','28.8','38.9','24.0']),
        'amortization_policy': (208, 3, ['0.006','0.013','0.020','0.007'], ['6.2','8.2','3.4','9.6'], ['7.2','6.2','11.5','2.9']),
        'always_graphs': (128, 3, ['0.038','0.068','0.049','0.050'], ['29.8','32.7','22.6','37.5'], ['0.0']*4),
        'static_threshold': (200, 3, ['0.026','0.034','0.055','0.026'], ['13.5','15.4','11.1','16.3'], ['7.2','6.2','11.5','2.9']),
        'static_gate_only': (192, 3, ['0.008','0.014','0.021','0.007'], ['6.2','8.2','3.4','9.6'], ['7.2','6.2','11.5','2.9']),
        'always_compile': (208, 2, ['49.15','89.25','48.94','30.93'], ['100.0']*4, ['0.0']*4),
        'first_k_probe_policy': (208, 3, ['0.000']*4, ['4.3','5.3','3.4','6.2'], ['4.3','4.3','3.8','2.9']),
        'risk_aware_policy': (208, 3, ['0.000','0.000','0.002','0.000'], ['5.8','8.7','5.8','6.2'], ['2.9','0.0','4.8','1.9']),
    }
    for policy, (valid, digits, p90, slow, missed) in policies.items():
        for i, device in enumerate(devices):
            value = report['controlled'][device]['cold']['policy_metrics'][policy]
            check(f'Table IV/{policy}/{device}',
                  (value['n_valid'], f"{value['p90_normalized_regret']:.{digits}f}",
                   f"{100 * value['slow_admit_rate']:.1f}", f"{100 * value['missed_opportunity_rate']:.1f}"),
                  (valid, p90[i], slow[i], missed[i]))

    queues = {
        'oracle_best_action': ([9.26,3.55,3.25,2.73], [1.60,.86,1.09,.87], [.88,.89,.89,.90]),
        'amortization_policy': ([9.55,3.61,3.32,2.75], [1.71,.87,1.12,.87], [.89,.89,.90,.90]),
        'always_eager': ([9.72,4.12,4.94,3.08], [1.72,1.01,1.38,.96], [.90,.92,.95,.92]),
        'always_graphs': ([9.72,3.72,3.38,2.84], [1.77,.92,1.13,.91], [.89,.90,.90,.90]),
        'static_threshold': ([176,243,323,244], [20,20,25,20], [2.08,2.84,2.49,2.15]),
        'always_compile': ([2648,3572,4732,3482], [28,28,35,28], [17.22,28.08,22.73,17.45]),
    }
    for policy, (means, sds, loads) in queues.items():
        digits = 0 if policy in ['static_threshold', 'always_compile'] else 2
        for i, device in enumerate(devices):
            value = report['controlled'][device]['queue_statistics'][policy]
            check(f'Table V/{policy}/{device}',
                  tuple(f'{value[field]:.{precision}f}' for field, precision in
                        [('mean', digits), ('sample_sd', digits), ('mean_load', 2)]),
                  (f'{means[i]:.{digits}f}', f'{sds[i]:.{digits}f}', f'{loads[i]:.2f}'))

    mixes = {
        'uniform': ['1.79','12.06–12.49','32.71','10.54'],
        'graph_friendly_heavy': ['5.59','26.48–27.33','57.04','18.60'],
        'dynamic_irregular_heavy': ['0.05','7.15–7.26','30.25','10.42'],
        'short_step_heavy': ['0.59','4.94–5.40','12.57','3.07'],
    }
    for mix, expected in mixes.items():
        for i, device in enumerate(devices):
            panels = [report['controlled'][device]] + ([report['replication']] if device == 'v100' else [])
            reductions = []
            for panel in panels:
                values = {r['policy']: r['p95_completion_time_s'] for r in panel['queue']
                          if r['job_mix_id'] == mix and r['load_level'] == 'heavy'}
                reductions.append(100 * (1 - values['static_gate_plus_eligibility'] / values['always_eager']))
            text = '–'.join(f'{value:.2f}' for value in sorted(reductions))
            check(f'Table VI/{mix}/{device}', text, expected[i])
    check('Source identity', read(results / 'validation.json')['paper_source_sha256'],
          '6bb6d447b3918cdcfa2bd97811132cef995f296b312d23e89d6078be7eb5ff78')
    result = {'passed': True, 'checks': checks, 'check_count': len(checks),
              'reference': '2026383989.pdf',
              'paper_pdf_sha256': '25df38dc10bf3ef8cbdefab225b44a1b203cc4f729e57243abef20df23a7f9d5',
              'scope': 'Table I-VI displayed values, Figure 2 cells, Figure 3 MobileNet estimates; full plot values checked by reproduce_camera_ready.py'}
    output = args.out or results / 'submitted_paper_verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'checks': len(checks)}, indent=2))


if __name__ == '__main__':
    main()
