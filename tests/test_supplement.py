"""CPU checks for the independent supplemental protocol, never GPU evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shortjob_runner.supplement_protocol import core_tasks, real_tasks, task_id, validate_single_gpu
from shortjob_runner.supplement_runtime import snapshot, compare, check_trajectory
from shortjob_runner.correctness import make_seeded_workload


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


if __name__ == '__main__':
    unittest.main()
