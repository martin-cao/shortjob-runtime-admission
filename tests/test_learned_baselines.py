from __future__ import annotations

import unittest

import numpy as np

from shortjob_runner.learned_baselines import (
    DecisionTreeClassifier,
    LogisticRegressionOVR,
    collapse_oracle_label,
    static_feature_vector,
)


class FeatureTests(unittest.TestCase):
    def test_static_features_are_workload_agnostic(self) -> None:
        row = {
            "workload_id": "anything",
            "num_steps": 100,
            "batch_size": 10,
            "shape_stability": "stable",
        }
        feats = static_feature_vector(row)
        # log10(100)=2, log10(10)=1, stable=1, mostly_stable=0
        self.assertEqual(len(feats), 4)
        self.assertAlmostEqual(feats[0], 2.0)
        self.assertAlmostEqual(feats[1], 1.0)
        self.assertEqual(feats[2], 1.0)
        self.assertEqual(feats[3], 0.0)

    def test_collapse_oracle_label(self) -> None:
        self.assertEqual(collapse_oracle_label("graphs_only"), "graphs")
        self.assertEqual(collapse_oracle_label("graphs_input_copy"), "graphs")
        self.assertEqual(collapse_oracle_label("compile_only"), "compile")
        self.assertEqual(collapse_oracle_label("compile_reduce_overhead"), "compile")
        self.assertEqual(collapse_oracle_label("best_eager"), "best_eager")
        self.assertEqual(collapse_oracle_label("eager"), "eager")
        self.assertEqual(collapse_oracle_label(None), "eager")


class LogisticRegressionTests(unittest.TestCase):
    def test_learns_separable_threshold(self) -> None:
        # Label depends only on the first feature (log steps): high -> graphs.
        feats = []
        labels = []
        for steps_log in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            feats.append([steps_log, 1.0, 1.0, 0.0])
            labels.append("graphs" if steps_log >= 1.5 else "eager")
        model = LogisticRegressionOVR().fit(np.array(feats), labels)
        self.assertEqual(model.predict_one([3.0, 1.0, 1.0, 0.0]), "graphs")
        self.assertEqual(model.predict_one([0.0, 1.0, 1.0, 0.0]), "eager")

    def test_single_class_training_predicts_that_class(self) -> None:
        feats = np.array([[1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]])
        model = LogisticRegressionOVR().fit(feats, ["eager", "eager"])
        self.assertEqual(model.predict_one([5.0, 5.0, 1.0, 1.0]), "eager")

    def test_deterministic(self) -> None:
        feats = np.array([[0.0, 1.0, 1.0, 0.0], [3.0, 1.0, 1.0, 0.0]])
        labels = ["eager", "graphs"]
        a = LogisticRegressionOVR().fit(feats, labels).predict_one([2.0, 1.0, 1.0, 0.0])
        b = LogisticRegressionOVR().fit(feats, labels).predict_one([2.0, 1.0, 1.0, 0.0])
        self.assertEqual(a, b)


class DecisionTreeTests(unittest.TestCase):
    def test_learns_threshold_split(self) -> None:
        feats = []
        labels = []
        for steps_log in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            feats.append([steps_log, 1.0, 1.0, 0.0])
            labels.append("graphs" if steps_log >= 2.0 else "eager")
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1).fit(np.array(feats), labels)
        self.assertEqual(tree.predict_one([3.5, 1.0, 1.0, 0.0]), "graphs")
        self.assertEqual(tree.predict_one([0.0, 1.0, 1.0, 0.0]), "eager")

    def test_single_class(self) -> None:
        feats = np.array([[1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]])
        tree = DecisionTreeClassifier().fit(feats, ["compile", "compile"])
        self.assertEqual(tree.predict_one([9.0, 9.0, 1.0, 1.0]), "compile")


if __name__ == "__main__":
    unittest.main()
