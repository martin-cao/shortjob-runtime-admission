from collections import defaultdict

EAGER = {'eager', 'best_eager', 'eager_inference_mode'}


def summarize_policies(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['policy']].append(row)
    summaries = {}
    for name, records in groups.items():
        valid = [r for r in records if not (r.get('policy_failed') or r.get('policy_correctness_failure'))]
        regrets = sorted(float(r['normalized_regret_vs_oracle']) for r in valid)
        slow = sum(r['admitted_action'] not in EAGER and
                   float(r['total_runtime_s']) > float(r['eager_baseline_total_runtime_s']) for r in valid)
        missed = sum(bool(r['missed_opportunity']) for r in valid)
        blocked = sum(bool(r.get('policy_correctness_failure')) for r in records)
        n = len(records)
        summaries[name] = {'n_conditions': n, 'n_valid': len(valid),
                           'n_correctness_blocked': blocked, 'n_other_failed': n - len(valid) - blocked,
                           'slow_admit_count': slow, 'slow_admit_rate': slow / n,
                           'missed_opportunity_count': missed, 'missed_opportunity_rate': missed / n,
                           'p90_normalized_regret': regrets[int(.9 * (len(regrets) - 1))] if regrets else None}
    return summaries
