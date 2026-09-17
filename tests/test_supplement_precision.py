"""Precision is part of the experiment identity and of correctness eligibility."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shortjob_runner.supplement_precision import configure_precision, with_precision, campaign_precision
from shortjob_runner.supplement_protocol import real_tasks, task_id
from shortjob_runner.supplement import passing_gate_ids

class PrecisionTests(unittest.TestCase):
    def test_strict_configuration_disables_tf32_without_changing_tolerances(self):
        before=(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            configure_precision('strict_fp32')
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(),'highest')
            configure_precision('legacy_cudnn_tf32')
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.set_float32_matmul_precision(before[0])
            torch.backends.cuda.matmul.allow_tf32=before[1]
            torch.backends.cudnn.allow_tf32=before[2]

    def test_new_precision_changes_identity_and_does_not_mutate_legacy_tasks(self):
        old=real_tasks(['resnet50'],lengths=[10],repeats=1)
        new=with_precision(old,'strict_fp32')
        self.assertTrue(all(t['protocol']=='supplement-v1' and 'precision_policy' not in t for t in old))
        self.assertTrue(all(t['protocol']=='supplement-v2' for t in new))
        self.assertTrue(all(task_id(a)!=task_id(b) for a,b in zip(old,new)))
        self.assertEqual(with_precision(old,'legacy_cudnn_tf32'),old)
        self.assertEqual(campaign_precision(new),('supplement-v2','strict_fp32'))

    def test_mixed_precision_or_bad_protocol_is_rejected(self):
        old=real_tasks(['resnet50'],lengths=[10],repeats=1)
        new=with_precision(old,'strict_fp32')
        with self.assertRaises(ValueError):campaign_precision([old[0],new[0]])
        with self.assertRaises(ValueError):campaign_precision([{**new[0],'protocol':'supplement-v1'}])
        with self.assertRaises(ValueError):configure_precision('typo')

    def test_legacy_pass_cannot_admit_strict_performance(self):
        old=[t for t in real_tasks(['resnet50'],lengths=[10],repeats=1) if t['action']=='eager']
        new=with_precision(old,'strict_fp32')
        timed=next(t for t in new if t['phase']=='performance')
        rows={task_id(t):dict(task_id=task_id(t),task=t,status='passed') for t in old if t['phase']=='correctness'}
        self.assertFalse(passing_gate_ids(timed,rows))
        rows={task_id(t):dict(task_id=task_id(t),task=t,status='passed') for t in new if t['phase']=='correctness'}
        self.assertEqual(len(passing_gate_ids(timed,rows)),2)

if __name__=='__main__':unittest.main()
