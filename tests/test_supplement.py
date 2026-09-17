"""CPU checks for the independent supplemental protocol, never GPU evidence."""
import json
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortjob_runner.supplement_protocol import core_tasks, real_tasks, task_id, validate_single_gpu
from shortjob_runner.supplement_runtime import snapshot, compare, check_trajectory
from shortjob_runner.correctness import make_seeded_workload
from shortjob_runner.supplement import passing_gate_ids, summarize, atomic_json, run_tasks, run_child
from shortjob_runner.supplement_runtime import Session


class ProtocolTests(unittest.TestCase):
    def test_core_has_unique_152_candidates_and_separate_controls(self):
        tasks = core_tasks(controls=False)
        self.assertEqual(len(tasks), 152)
        self.assertEqual(len({task_id(t) for t in tasks}), 152)
        self.assertTrue(all(t['checkpoints'][-1] == t['steps'] for t in tasks))
        self.assertGreater(len(core_tasks()), 152)

    def test_real_checks_cover_every_timed_condition(self):
        tasks = real_tasks(['resnet50', 'vit_b_16'], batch=1, repeats=5)
        checks = {(t['workload'], t['action'], t['steps'], t['seed'])
                  for t in tasks if t['phase'] == 'correctness'}
        timed = [t for t in tasks if t['phase'] == 'performance']
        self.assertEqual(len(timed), 120)
        self.assertTrue(all((t['workload'], t['action'], t['steps'], t['seed']) in checks for t in timed))
        self.assertEqual(len(tasks), len({task_id(t) for t in tasks}))

    def test_single_gpu_contract_rejects_multiple_or_absent_devices(self):
        for count in [0, 2, 4]:
            with self.assertRaises(ValueError):
                validate_single_gpu(count)
        validate_single_gpu(1)

    def test_gate_rejects_missing_seed_or_failure_on_other_seed(self):
        tasks = real_tasks(['resnet50'], lengths=[10], repeats=1)
        timed = next(t for t in tasks if t['phase'] == 'performance')
        gates = [t for t in tasks if t['phase'] == 'correctness' and t['action'] == timed['action']]
        rows = {task_id(t): {'task_id': task_id(t), 'task': t, 'status': 'passed'} for t in gates}
        self.assertEqual(len(passing_gate_ids(timed, rows)), 2)
        rows[task_id(gates[0])]['status'] = 'failed'
        self.assertFalse(passing_gate_ids(timed, rows))
        rows.pop(task_id(gates[0]))
        self.assertFalse(passing_gate_ids(timed, rows))

    def test_partial_summary_never_claims_completion(self):
        self.assertFalse(summarize(core_tasks(), {})['complete'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows' / 'x.json'
            atomic_json(path, {'status': 'failed'})
            self.assertEqual(json.loads(path.read_text())['status'], 'failed')
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_incomplete_performance_never_gets_a_speedup(self):
        tasks = real_tasks(['resnet50'], lengths=[10], repeats=2)
        selected = [t for t in tasks if t['phase'] == 'performance' and t['repeat'] == 0]
        rows = {task_id(t): dict(task=t, status='measured', total_s=1., initialization_s=.2, execution_s=.8) for t in selected}
        summary = summarize(tasks, rows)
        self.assertTrue(all(r['speedup_vs_eager'] is None for r in summary['performance_summary']))

    def test_campaign_resume_skips_completed_rows_and_rejects_changed_host(self):
        tasks = [t for t in real_tasks(['resnet50'], lengths=[10], repeats=1) if t['action'] == 'eager']
        env = {'runtime': {'torch': 'fixture'}, 'machine': {'hostname': 'test'},
               'gpu_identity': 'fixture', 'cuda_visible_devices': '0'}
        calls = []
        def child(command, child_env, log, timeout):
            t = json.loads(command[command.index('--task-json') + 1])
            output = Path(command[command.index('--output') + 1])
            row = {'task_id': task_id(t), 'task': t, 'status': 'passed'}
            if t['phase'] == 'performance':
                row.update(status='measured', initialization_s=1, execution_s=2, total_s=3)
            atomic_json(output, row)
            calls.append(t)
            return 0
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(platform='4060', suite='core', run_dir=Path(directory)/'run',
                                      assets=None, models=[], timeout=3, max_wall_seconds=None)
            with patch('shortjob_runner.supplement.environment', return_value=env), \
                 patch('shortjob_runner.supplement.run_child', side_effect=child), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_tasks(args, tasks), 0)
                self.assertEqual(run_tasks(args, tasks), 0)
                self.assertEqual(len(calls), 3)
                env['machine']['hostname'] = 'different-host'
                with self.assertRaises(ValueError):
                    run_tasks(args, tasks)

    def test_timeout_terminates_child_process(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_child([sys.executable, '-c', 'import time; time.sleep(30)'], dict(os.environ), log, 0.05)


class StateTests(unittest.TestCase):
    def test_snapshots_do_not_alias_and_include_gradient(self):
        workload = make_seeded_workload('short_train_small', 1, torch.device('cpu'), 42)
        workload.step()
        saved = snapshot(workload)
        self.assertTrue(any(k.startswith('grad:') for k in saved))
        original = {k: v.clone() for k, v in saved.items()}
        workload.step()
        for key in saved:
            torch.testing.assert_close(saved[key], original[key])
        self.assertFalse(compare(saved, snapshot(workload))['passed'])

    def test_reports_all_bad_keys_and_nonfinite_values(self):
        ref = {'a': torch.tensor([1.]), 'b': torch.tensor([2.])}
        got = {'a': torch.tensor([float('nan')]), 'b': torch.tensor([8.])}
        result = compare(ref, got)
        self.assertFalse(result['passed'])
        self.assertEqual(set(result['errors']), {'a', 'b'})
        json.dumps(result, allow_nan=False)

    def test_cpu_training_matches_exact_n_updates(self):
        row = check_trajectory('short_train_small', 'eager', 1, 42, [1, 3, 10], torch.device('cpu'))
        self.assertEqual(row['status'], 'passed')
        self.assertEqual([x['step'] for x in row['checks']], [1, 3, 10])
        self.assertTrue(row['checks'][-1]['comparison']['passed'])

    def test_refreshing_input_is_compared_against_matching_reference(self):
        row = check_trajectory('fixed_shape_infer_small', 'eager', 1, 17, [1, 3], torch.device('cpu'), refresh=True)
        self.assertEqual(row['status'], 'passed')

    def test_stale_input_candidate_fails_at_first_checkpoint(self):
        advance = Session.advance
        def stale(session, refresh):
            advance(session, False if session.action == 'best_eager' else refresh)
        with patch.object(Session, 'advance', stale):
            row = check_trajectory('fixed_shape_infer_small', 'best_eager', 1, 17, [1, 3], torch.device('cpu'), refresh=True)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['first_failing_checkpoint'], {'job': 1, 'step': 1})

    def test_extra_training_update_is_detected(self):
        def factory(w, b, d, s):
            result = make_seeded_workload(w, b, d, s)
            factory.calls += 1
            if factory.calls == 2:
                result.step()
            return result
        factory.calls = 0
        row = check_trajectory('short_train_small', 'eager', 1, 42, [1, 3], torch.device('cpu'), factory=factory)
        self.assertEqual(row['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
