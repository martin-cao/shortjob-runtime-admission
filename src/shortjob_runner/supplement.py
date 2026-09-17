"""Local, sequential single-GPU supplement runner with immutable run manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback

from .supplement_protocol import MODELS, PROTOCOL, core_tasks, real_tasks, task_id, validate_single_gpu


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(platform_name):
    import torch
    from .check_env import collect_environment, validate_environment
    from .env import machine_info, nvidia_smi_snapshot

    validate_single_gpu(torch.cuda.device_count())
    env = collect_environment()
    try:
        env['torchvision'] = importlib.metadata.version('torchvision')
    except importlib.metadata.PackageNotFoundError:
        env['torchvision'] = None
    problems = validate_environment(env)
    if platform_name not in str(env.get('gpu', '')).lower():
        problems.append(f'Platform {platform_name} does not match {env.get("gpu")}')
    if problems:
        raise ValueError('; '.join(problems))
    machine = machine_info()
    machine.pop('pid', None)
    machine['cpu_model'] = next((s.split(':', 1)[1].strip() for s in (read_optional('/proc/cpuinfo') or '').splitlines()
                                  if s.startswith('model name')), None)
    machine['affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    machine['cgroup_membership'] = read_optional('/proc/self/cgroup')
    machine['cgroup_cpu_max_root'] = read_optional('/sys/fs/cgroup/cpu.max')
    machine['cgroup_memory_max_root'] = read_optional('/sys/fs/cgroup/memory.max')
    machine['meminfo'] = read_optional('/proc/meminfo')
    try:
        identity = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        identity = None
    return {'runtime': env, 'machine': machine, 'gpu_identity': identity,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'nvidia_smi': nvidia_smi_snapshot()}


def resume_identity(env):
    # Variable temperature/utilization/free memory must not prevent a same-host resume.
    machine = {k: v for k, v in env['machine'].items() if k != 'meminfo'}
    return {'runtime': env['runtime'], 'machine': machine,
            'gpu_identity': env['gpu_identity'], 'cuda_visible_devices': env['cuda_visible_devices']}


def gate_key(task):
    return tuple(task[k] for k in ('workload', 'action', 'batch', 'steps'))


def passing_gate_ids(task, results):
    gates = [r for r in results.values() if r['task']['phase'] == 'correctness'
             and gate_key(r['task']) == gate_key(task)]
    if {r['task']['seed'] for r in gates} != {17, 42} or any(r['status'] != 'passed' for r in gates):
        return []
    return [r['task_id'] for r in gates]


def run_child(command, env, log, timeout):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # Stop only this experiment's process group, including compiler workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def worker(task, assets):
    import torch
    from .env import runtime_info, nvidia_smi_snapshot
    from .supplement_runtime import check_trajectory, measure

    validate_single_gpu(torch.cuda.device_count())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    torch.cuda.init()
    before = nvidia_smi_snapshot()
    factory = None
    if task['kind'] == 'real_images':
        from .supplement_models import RealImages, validate_assets
        validate_assets(assets, task['workload'])
        factory = lambda w, b, d, s: RealImages(w, b, d, s, assets)
    if task['phase'] == 'correctness':
        result = check_trajectory(task['workload'], task['action'], task['batch'], task['seed'],
                                  task['checkpoints'], device, jobs=task['jobs'], factory=factory,
                                  refresh=True if task['kind'] == 'real_images' else None)
    else:
        result = measure(task['workload'], task['action'], task['batch'], task['seed'],
                         task['steps'], device, factory)
        if not result['final_state_finite']:
            result.update(status='failed', failure_stage='nonfinite_after_measurement')
    return {**result, 'runtime': runtime_info(), 'gpu_before': before, 'gpu_after': nvidia_smi_snapshot()}


def run_tasks(args, tasks):
    from .supplement_models import sha256, validate_assets

    if not args.platform or not args.run_dir:
        raise ValueError('--platform and --run-dir are required for execution')
    if args.suite != 'core' and args.assets is None:
        raise ValueError('--assets is required for real models')
    assets = args.assets.resolve() if args.assets else None
    if assets:
        for model in args.models:
            validate_assets(assets, model)
    env = environment(args.platform)
    manifest = {'protocol': PROTOCOL, 'platform': args.platform, 'tasks': tasks,
                'source_hash': source_hash(), 'environment_identity': resume_identity(env),
                'assets_manifest_hash': sha256(assets / 'manifest.json') if assets else None,
                'timeout_seconds': args.timeout}
    folder = args.run_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Run manifest differs (code, environment, assets or tasks); use a new run directory')
    else:
        if any(folder.iterdir()):
            raise ValueError('Run directory is not empty and has no manifest')
        atomic_json(manifest_path, manifest)
        atomic_json(folder / 'environment.json', env)
    # One writer per run; OS releases the lock if interrupted.
    import fcntl
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for path in (folder / 'rows').glob('*.json'):
            row = json.loads(path.read_text())
            if row['task_id'] != path.stem or row['task_id'] != task_id(row['task']) or row['task_id'] not in {task_id(t) for t in tasks}:
                raise ValueError('Unexpected result row in run directory')
            results[row['task_id']] = row
        started = time.monotonic()
        rng = random.Random(20260917)
        checks = [t for t in tasks if t['phase'] == 'correctness']
        timed = [t for t in tasks if t['phase'] == 'performance']
        rng.shuffle(checks)
        rng.shuffle(timed)
        ordered = checks + sorted(timed, key=lambda t: t['repeat'])
        for task in ordered:
            ident = task_id(task)
            if ident in results:
                continue
            if args.max_wall_seconds and time.monotonic()-started >= args.max_wall_seconds:
                break
            row = {'task_id': ident, 'task': task}
            if task['phase'] == 'performance':
                gate_ids = passing_gate_ids(task, results)
                if not gate_ids:
                    row.update(status='blocked', failure_stage='correctness_gate')
                else:
                    row['gate_task_ids'] = gate_ids
            output = folder / 'rows' / f'{ident}.json'
            if 'status' not in row:
                cache = folder / 'cache' / ident
                # An interrupted attempt must not turn the next attempt into a warm-cache run.
                attempt = cache / str(time.time_ns())
                attempt.mkdir(parents=True)
                child_env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                             'TORCHINDUCTOR_CACHE_DIR': str(attempt / 'inductor'),
                             'TRITON_CACHE_DIR': str(attempt / 'triton'), 'CUDA_CACHE_PATH': str(attempt / 'cuda')}
                command = [sys.executable, '-m', 'shortjob_runner.supplement', '_worker',
                           '--task-json', json.dumps(task), '--output', str(output)]
                if assets:
                    command += ['--assets', str(assets)]
                logs = folder / 'logs'
                logs.mkdir(exist_ok=True)
                try:
                    with (logs / f'{ident}.log').open('a') as stream:
                        returncode = run_child(command, child_env, stream, args.timeout)
                    if output.exists():
                        child = json.loads(output.read_text())
                        row.update(child)
                        if returncode != 0 and row.get('status') in {'passed', 'measured'}:
                            row.update(status='failed', failure_stage='worker_exit', returncode=returncode)
                    else:
                        row.update(status='failed', failure_stage='worker_process', returncode=returncode)
                except subprocess.TimeoutExpired:
                    row.update(status='timeout', failure_stage='worker_timeout')
            atomic_json(output, row)
            results[ident] = row
            print(json.dumps({'finished': len(results), 'planned': len(tasks), 'task_id': ident, 'status': row['status']}), flush=True)
        summary = summarize(tasks, results)
        atomic_json(folder / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return 0 if summary['complete'] and not summary['nonpassing'] else 1


def summarize(tasks, results):
    counts = Counter(row['status'] for row in results.values())
    nonpassing = sum(r['status'] not in {'passed', 'measured'}
                     and not (r['task']['kind'] == 'precheck_control' and r['status'] == 'infeasible')
                     for r in results.values())
    eager = [dict(workload=r['task']['workload'], batch=r['task']['batch'], steps=r['task']['steps'],
                  repeat=r['task']['repeat'], initialization_s=r['initialization_s'], execution_s=r['execution_s'], total_s=r['total_s'])
             for r in results.values() if r['task']['phase'] == 'performance'
             and r['task']['action'] == 'eager' and r['status'] == 'measured']
    expected = Counter((t['workload'], t['batch'], t['steps'], t['action'])
                       for t in tasks if t['phase'] == 'performance')
    buckets = defaultdict(list)
    for row in results.values():
        if row['status'] == 'measured' and row['task']['phase'] == 'performance':
            t = row['task']
            buckets[(t['workload'], t['batch'], t['steps'], t['action'])].append(row['total_s'])
    performance = []
    for key, planned in sorted(expected.items()):
        values = buckets[key]
        baseline = buckets[(*key[:3], 'eager')]
        complete = len(values) == planned
        baseline_complete = len(baseline) == expected[(*key[:3], 'eager')] and bool(baseline)
        performance.append(dict(workload=key[0], batch=key[1], steps=key[2], action=key[3],
                                repeats=len(values), planned_repeats=planned, complete=complete,
                                median_total_s=statistics.median(values) if values else None,
                                speedup_vs_eager=statistics.median(baseline)/statistics.median(values)
                                if complete and baseline_complete and min(values) > 0 else None))
    return {'planned': len(tasks), 'finished': len(results), 'complete': len(results) == len(tasks),
            'counts': dict(counts), 'nonpassing': nonpassing,
            'performance_summary': performance,
            'eager_duration_observations': eager,
            'short_job_qualification': 'pending_protocol_review_of_eager_durations; no automatic model-level claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Prepare real images and download explicit pretrained weights on CPU')
    prepare.add_argument('--images', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--models', nargs='+', choices=MODELS, default=['resnet50', 'vit_b_16'])
    prepare.add_argument('--limit', type=int, default=64)
    run = commands.add_parser('run', help='Defaults to a dry plan; --execute launches CUDA subprocesses')
    run.add_argument('--suite', choices=['core', 'characterize', 'real'], default='core')
    run.add_argument('--models', nargs='+', choices=MODELS, default=['resnet50', 'vit_b_16'])
    run.add_argument('--platform', choices=['4060', 'v100', 'a100', 'h100'])
    run.add_argument('--batch', type=int, default=1)
    run.add_argument('--lengths', type=int, nargs='+', default=[10, 50, 100, 500])
    run.add_argument('--repeats', type=int, default=5)
    run.add_argument('--assets', type=Path)
    run.add_argument('--run-dir', type=Path)
    run.add_argument('--timeout', type=float, default=900)
    run.add_argument('--max-wall-seconds', type=float)
    run.add_argument('--execute', action='store_true')
    child = commands.add_parser('_worker')
    child.add_argument('--task-json', required=True)
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--assets', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .supplement_models import prepare_assets
        prepare_assets(args.images, args.out, args.models, args.source, args.limit)
        return 0
    if args.command == '_worker':
        task = json.loads(args.task_json)
        row = {'task_id': task_id(task), 'task': task}
        try:
            row.update(worker(task, args.assets))
        except Exception as exc:
            row.update(status='infeasible' if str(exc).startswith('precheck:') else 'failed',
                       failure_stage='precheck' if str(exc).startswith('precheck:') else 'execution',
                       error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        atomic_json(args.output, row)
        return 0 if row['status'] in {'passed', 'measured'} else 1
    if args.timeout <= 0 or (args.max_wall_seconds is not None and args.max_wall_seconds <= 0):
        parser.error('Timeout and optional wall budget must be positive')
    tasks = core_tasks() if args.suite == 'core' else real_tasks(args.models, batch=args.batch, repeats=args.repeats, lengths=args.lengths)
    if args.suite == 'characterize':
        tasks = [t for t in tasks if t['action'] == 'eager']
    if not args.execute:
        print(json.dumps({'protocol': PROTOCOL, 'planned': len(tasks), 'phases': dict(Counter(t['phase'] for t in tasks)),
                          'candidate_models_not_yet_qualified_as_short_jobs': args.models if args.suite != 'core' else [],
                          'tasks': tasks}, indent=2))
        return 0
    return run_tasks(args, tasks)


if __name__ == '__main__':
    raise SystemExit(main())
