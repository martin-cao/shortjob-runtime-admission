"""Workload-agnostic learned admission baselines (pure numpy, no new deps).

These models exist to answer a single reviewer question: can an estimator that
sees *only* workload-agnostic static features (job length, batch size, shape
stability) — and never the workload id — recommend a runtime action that
generalizes to a workload family it was never calibrated on?

They are evaluated leave-one-workload-out by the policy layer.  If they still
collapse to ``eager`` on held-out families, that confirms the decision-support
scope limitation; if they recover real graph/compile opportunities, that
strengthens the contribution.  Either outcome is reported honestly.

Implementation notes:
- Pure numpy so the artifact runs on every recorded environment (holger_sc3
  included) and stays bit-reproducible across machines.  No scikit-learn / scipy.
- Deterministic: logistic regression uses full-batch gradient descent from a
  zero initialization; the decision tree is a greedy CART with fixed tie-breaks.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# Collapsed action label space the learners predict over.  The summary oracle is
# already correctness-constrained, so labels never point at a gated action.
ACTION_LABELS = ("eager", "best_eager", "graphs", "compile")

# Map a predicted label to a concrete action id; eligibility is checked later.
LABEL_TO_ACTION = {
    "eager": "eager",
    "best_eager": "best_eager",
    "graphs": "graphs_only",
    "compile": "compile_only",
}

_GRAPH_ACTIONS = {
    "graphs_only",
    "graphs_input_copy",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
}
_COMPILE_ACTIONS = {
    "compile_only",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
}


def static_feature_vector(row: dict[str, Any]) -> list[float]:
    """Workload-agnostic features only.  No workload id, family, or runtimes.

    A reviewer must be able to confirm by inspection that nothing workload- or
    measurement-specific leaks in: the vector is derived purely from the job's
    declared static metadata.
    """
    steps = float(row.get("num_steps") or 1)
    batch = float(row.get("batch_size") or 1)
    stability = row.get("shape_stability")
    return [
        math.log10(max(steps, 1.0)),
        math.log10(max(batch, 1.0)),
        1.0 if stability == "stable" else 0.0,
        1.0 if stability == "mostly_stable" else 0.0,
    ]


def collapse_oracle_label(oracle_action: str | None) -> str:
    if oracle_action in _GRAPH_ACTIONS:
        return "graphs"
    if oracle_action in _COMPILE_ACTIONS:
        return "compile"
    if oracle_action == "best_eager":
        return "best_eager"
    return "eager"


class LogisticRegressionOVR:
    """One-vs-rest multinomial-ish logistic regression via full-batch GD.

    Deterministic: zero init, fixed iterations and learning rate, standardized
    features.  Returns the existing-class argmax; classes absent from training
    are never predicted.
    """

    def __init__(self, *, iters: int = 500, lr: float = 0.1, l2: float = 1e-3) -> None:
        self.iters = iters
        self.lr = lr
        self.l2 = l2
        self._classes: list[str] = []
        self._weights: np.ndarray | None = None  # (n_classes, n_features + 1)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._fallback: str = "eager"

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> "LogisticRegressionOVR":
        labels = list(labels)
        self._classes = sorted(set(labels), key=ACTION_LABELS.index)
        if not self._classes:
            return self
        self._fallback = max(self._classes, key=lambda c: labels.count(c))
        if len(self._classes) == 1:
            self._weights = None
            return self
        self._mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        self._std = std
        x = (features - self._mean) / self._std
        x = np.hstack([x, np.ones((x.shape[0], 1))])  # bias column
        n_classes = len(self._classes)
        weights = np.zeros((n_classes, x.shape[1]))
        for ci, cls in enumerate(self._classes):
            y = np.array([1.0 if lbl == cls else 0.0 for lbl in labels])
            w = np.zeros(x.shape[1])
            for _ in range(self.iters):
                z = x @ w
                pred = 1.0 / (1.0 + np.exp(-z))
                grad = x.T @ (pred - y) / x.shape[0] + self.l2 * w
                w -= self.lr * grad
            weights[ci] = w
        self._weights = weights
        return self

    def predict_one(self, feature: list[float]) -> str:
        if not self._classes:
            return "eager"
        if self._weights is None or self._mean is None or self._std is None:
            return self._fallback
        x = (np.array(feature) - self._mean) / self._std
        x = np.append(x, 1.0)
        scores = self._weights @ x
        return self._classes[int(np.argmax(scores))]


class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "label")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: "_TreeNode | None" = None
        self.right: "_TreeNode | None" = None
        self.label: str | None = None


class DecisionTreeClassifier:
    """Depth-limited greedy CART (Gini), deterministic tie-breaks.

    Kept shallow on purpose: a small, inspectable tree is the point — it shows
    whether a few static thresholds suffice, not whether a large model can
    memorize the matrix.
    """

    def __init__(self, *, max_depth: int = 3, min_samples_leaf: int = 4) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._root: _TreeNode | None = None
        self._fallback = "eager"

    @staticmethod
    def _gini(labels: list[str]) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        impurity = 1.0
        for cls in set(labels):
            p = labels.count(cls) / n
            impurity -= p * p
        return impurity

    def _majority(self, labels: list[str]) -> str:
        if not labels:
            return self._fallback
        # Tie-break by ACTION_LABELS order for determinism.
        return max(sorted(set(labels), key=ACTION_LABELS.index), key=labels.count)

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> "DecisionTreeClassifier":
        labels = list(labels)
        if labels:
            self._fallback = self._majority(labels)
        self._root = self._build(features, labels, depth=0)
        return self

    def _build(self, x: np.ndarray, labels: list[str], *, depth: int) -> _TreeNode:
        node = _TreeNode()
        if (
            depth >= self.max_depth
            or len(labels) < 2 * self.min_samples_leaf
            or len(set(labels)) <= 1
        ):
            node.label = self._majority(labels)
            return node

        best = None  # (gini, feature, threshold, left_idx, right_idx)
        n_features = x.shape[1]
        for feature in range(n_features):
            values = sorted(set(x[:, feature].tolist()))
            for a, b in zip(values, values[1:]):
                threshold = (a + b) / 2.0
                left_mask = x[:, feature] <= threshold
                left_labels = [lbl for lbl, m in zip(labels, left_mask) if m]
                right_labels = [lbl for lbl, m in zip(labels, left_mask) if not m]
                if len(left_labels) < self.min_samples_leaf or len(right_labels) < self.min_samples_leaf:
                    continue
                n = len(labels)
                weighted = (
                    len(left_labels) / n * self._gini(left_labels)
                    + len(right_labels) / n * self._gini(right_labels)
                )
                candidate = (weighted, feature, threshold)
                if best is None or candidate < best[:3]:
                    best = (weighted, feature, threshold, left_mask)

        if best is None:
            node.label = self._majority(labels)
            return node

        _, feature, threshold, left_mask = best
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(
            x[left_mask], [lbl for lbl, m in zip(labels, left_mask) if m], depth=depth + 1
        )
        node.right = self._build(
            x[~left_mask], [lbl for lbl, m in zip(labels, left_mask) if not m], depth=depth + 1
        )
        return node

    def predict_one(self, feature: list[float]) -> str:
        node = self._root
        if node is None:
            return self._fallback
        while node.label is None:
            assert node.feature is not None and node.threshold is not None
            node = node.left if feature[node.feature] <= node.threshold else node.right
            if node is None:
                return self._fallback
        return node.label


def build_model(kind: str):
    if kind == "logreg":
        return LogisticRegressionOVR()
    if kind == "tree":
        return DecisionTreeClassifier()
    raise ValueError(f"unknown learned-baseline kind: {kind!r}")
