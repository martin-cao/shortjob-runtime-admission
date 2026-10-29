import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'03_src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from legacy_reanalysis import prepare_summaries, evaluate_variant


def sample(action,time,steps=10):
    return dict(workload_id='toy',task_family='inference',shape_stability='stable',
                batch_size=1,num_steps=steps,action=action,action_feasible=True,
                median_total_runtime_s=time,n_repeats=5,median_graph_capture_overhead_s=0,
                median_avg_step_time_ms=time/steps*1000)


class MatchedCameraReadyTests(unittest.TestCase):
    def test_baseline_is_predetermined_even_when_literal_eager_is_faster(self):
        source=[sample('eager',1),sample('best_eager',2),sample('graphs_only',1.5),sample('graphs_input_copy',.01)]
        original=copy.deepcopy(source)
        rows=prepare_summaries(source,'matched_eager',frozenset())
        self.assertEqual(source,original)
        eager=next(r for r in rows if r['action']=='eager')
        self.assertEqual(eager['measured_source_action'],'best_eager')
        self.assertEqual(eager['median_total_runtime_s'],2)
        self.assertEqual(eager['oracle_action'],'graphs_only')
        self.assertNotIn('graphs_input_copy',[r['action'] for r in rows])

    def test_new_analysis_does_not_unblock_historical_graph_failure(self):
        blocked=frozenset({('toy','graphs_only')})
        rows=prepare_summaries([sample('eager',2),sample('graphs_only',.01)],'matched_eager',blocked)
        self.assertTrue(all(r['oracle_action']=='eager' for r in rows))

    def test_target_timing_does_not_leak_into_calibration(self):
        source=[sample(a,t,n) for n in [10,100] for a,t in [('eager',3),('best_eager',2),('graphs_only',1)]]
        def choice(data):
            rows=prepare_summaries(data,'matched_eager',frozenset())
            return next(r['admitted_action'] for r in evaluate_variant(rows,'matched_eager',frozenset(),['amortization_policy']) if r['num_steps']==100)
        before=choice(source)
        source[-1]['median_total_runtime_s']=10000
        self.assertEqual(choice(source),before)
