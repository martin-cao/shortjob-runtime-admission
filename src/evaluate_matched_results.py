import argparse
import hashlib
import json
from pathlib import Path

from legacy_reanalysis import evaluate_variant, normalized_trace, prepare_summaries, queue_on_trace
from paper_metrics import summarize_policies
from shortjob_runner.summarize import load_correctness_blocklist


def main():
    ap = argparse.ArgumentParser(description='从测量汇总复现 matched-eager 策略与队列分析')
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--correctness-gate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are rejected')
    ap.add_argument('--num-jobs', type=int, default=1000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--loads', type=float, nargs='+', default=[.35, .65, .9])
    ap.add_argument('--workload-weights', type=Path, help='Optional JSON mapping workload ID to weight')
    ap.add_argument('--step-weights', type=Path, help='Optional JSON mapping horizon to weight')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Queue seeds must be unique')
    inputs = [args.summary, args.correctness_gate] + [p for p in [args.workload_weights,args.step_weights] if p]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    read = lambda p: json.loads(p.read_text()) if p else None
    ww, sw = read(args.workload_weights), read(args.step_weights)
    if sw is not None: sw = {int(k): v for k,v in sw.items()}
    raw = [json.loads(line) for line in args.summary.read_text().splitlines() if line.strip()]
    blocked = load_correctness_blocklist(args.correctness_gate)
    rows = prepare_summaries(raw, 'matched_eager', blocked)
    policies = evaluate_variant(rows, 'matched_eager', blocked)
    queues = []
    for load in args.loads:
        for seed in args.seeds:
            rate, trace, mean_s = normalized_trace(rows,args.num_jobs,seed,load,ww,sw)
            queues.extend(queue_on_trace(rows,policies,trace,rate,blocked,{
                'arrival_basis':'revised_mix_specific_oracle_reference',
                'target_utilization':load,'queue_seed':seed,'normalization_oracle_mean_s':mean_s}))
    args.out.mkdir(parents=True)
    for name,data in [('action_summary',rows),('policy_eval',policies),('queue_replay',queues)]:
        (args.out/f'{name}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in data))
    (args.out/'policy_aggregate.json').write_text(json.dumps(summarize_policies(policies),indent=2,allow_nan=False)+'\n')
    manifest = {'variant':'matched_eager','input_sha256':hashes,'num_jobs':args.num_jobs,
                'seeds':args.seeds,'loads':args.loads,'workload_weights':ww,'step_weights':sw,
                'failed_action_recovery_cost_s':0,
                'scope':'Historical fixed-input summaries only; not supplemental input-refresh or grouped-reuse rows.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Wrote matched analysis to {args.out}; no GPU measurements or network requests.')


if __name__=='__main__': main()
