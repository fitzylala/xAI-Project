import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from analysis.structural import accuracy_by_depth, structural_metrics


def fit(X, y, **kwargs):
    return DecisionTreeClassifier(random_state=0, **kwargs).fit(np.array(X), np.array(y))


def test_single_split_tree_metrics():
    metrics = structural_metrics(fit([[0], [1], [2], [3]], [0, 0, 1, 1]))

    assert metrics == {
        "depth": 1,
        "n_leaves": 2,
        "n_nodes": 3,
        "n_features_used": 1,
        "mean_path_length": 1.0,
    }


def test_mean_path_length_weights_leaves_by_samples():
    # One sample is classified after 1 split, the other two after 2
    metrics = structural_metrics(fit([[0], [1], [2]], [0, 1, 0]))

    assert metrics["depth"] == 2
    assert metrics["mean_path_length"] == pytest.approx(5 / 3)


def test_features_used_counts_distinct_features():
    xor = fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])

    assert xor.tree_.node_count == 7  # three splits over two features
    assert structural_metrics(xor)["n_features_used"] == 2


def test_accuracy_by_depth_improves_with_budget():
    X = np.array([[i] for i in range(8)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    accuracy = accuracy_by_depth(X, X, y, y, criterion="gini", depths=(1, 7), random_state=0)

    assert set(accuracy) == {1, 7}
    assert accuracy[1] < 1.0
    assert accuracy[7] == 1.0
