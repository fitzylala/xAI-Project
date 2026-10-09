"""
Structural complexity metrics for a fitted decision tree.

Smaller, shallower trees that use fewer features are easier for a human to
trace through. See project-work/skye/idea-proposal.md, Phase 3.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF


def _mean_path_length(tree):
    """Average number of splits a training sample passes before reaching a leaf."""
    inner = tree.tree_
    total = 0
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if inner.children_left[node] == TREE_LEAF:
            total += depth * inner.n_node_samples[node]
        else:
            stack.append((inner.children_left[node], depth + 1))
            stack.append((inner.children_right[node], depth + 1))
    return float(total / inner.n_node_samples[0])


def structural_metrics(tree):
    """
    Measure the size and shape of a fitted tree.

    Args:
        tree: Fitted DecisionTreeClassifier

    Returns:
        dict with depth, n_leaves, n_nodes, n_features_used and
        mean_path_length (splits needed to classify an average training sample)
    """
    used = tree.tree_.feature
    return {
        "depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "n_nodes": int(tree.tree_.node_count),
        "n_features_used": len(np.unique(used[used >= 0])),
        "mean_path_length": _mean_path_length(tree),
    }


def accuracy_by_depth(X_train, X_test, y_train, y_test, criterion, depths, random_state=42):
    """
    Test accuracy of a tree at each depth limit (equal "complexity budgets").

    Args:
        X_train, X_test: Feature arrays (preprocessed)
        y_train, y_test: Target labels
        criterion: 'gini' or 'entropy'
        depths: Depth limits to train at, e.g. (3, 5, 7, 10)
        random_state: Random seed for reproducibility

    Returns:
        dict mapping each depth to its test accuracy
    """
    accuracy = {}
    for depth in depths:
        tree = DecisionTreeClassifier(
            criterion=criterion, max_depth=depth, random_state=random_state
        )
        tree.fit(X_train, y_train)
        accuracy[depth] = float(tree.score(X_test, y_test))
    return accuracy
