"""
Structural comparison of two fitted decision trees.

Used to check whether the Gini and Entropy trees are the same tree before any
interpretability comparison, and, when they are not, to show where they differ
in the part of the tree a human would actually read (the top levels).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.tree._tree import TREE_LEAF


@dataclass
class NodeDifference:
    path: str   # e.g. "root > left > right"
    depth: int  # root is depth 0
    kind: str   # "feature", "threshold", "shape" or "leaf_class"
    a: str      # what tree_a has at this position
    b: str      # what tree_b has at this position


@dataclass
class TreeComparison:
    max_depth: int
    differences: list[NodeDifference] = field(default_factory=list)

    @property
    def identical_within_depth(self):
        return not self.differences

    @property
    def first_divergence_depth(self):
        return self.differences[0].depth if self.differences else None


def _is_leaf(tree, node):
    return tree.tree_.children_left[node] == TREE_LEAF


def _leaf_class(tree, node):
    return tree.classes_[np.argmax(tree.tree_.value[node])]


def _describe(tree, node, feature_names):
    if _is_leaf(tree, node):
        return f"leaf (class {_leaf_class(tree, node)})"
    feature = tree.tree_.feature[node]
    name = feature_names[feature] if feature_names is not None else f"feature_{feature}"
    return f"{name} <= {tree.tree_.threshold[node]:.3f}"


def _diff(tree_a, tree_b, max_depth, feature_names, tol):
    """Walk both trees from the root in parallel, breadth first.

    Nodes are matched by position (the path of left/right turns from the root),
    not by node id. Only nodes at depth < max_depth are compared; None means
    the whole tree. Returns differences ordered by depth.
    """
    differences = []
    queue = deque([(0, 0, 0, "root")])

    while queue:
        node_a, node_b, depth, path = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue

        leaf_a, leaf_b = _is_leaf(tree_a, node_a), _is_leaf(tree_b, node_b)
        kind = None
        descend = False

        if leaf_a and leaf_b:
            if _leaf_class(tree_a, node_a) != _leaf_class(tree_b, node_b):
                kind = "leaf_class"
        elif leaf_a or leaf_b:
            kind = "shape"
        elif tree_a.tree_.feature[node_a] != tree_b.tree_.feature[node_b]:
            # Subtrees under different features are no longer comparable
            kind = "feature"
        else:
            descend = True
            gap = abs(tree_a.tree_.threshold[node_a] - tree_b.tree_.threshold[node_b])
            if gap > tol:
                kind = "threshold"

        if kind is not None:
            differences.append(NodeDifference(
                path=path, depth=depth, kind=kind,
                a=_describe(tree_a, node_a, feature_names),
                b=_describe(tree_b, node_b, feature_names),
            ))

        if descend:
            for side, children in (("left", "children_left"), ("right", "children_right")):
                queue.append((
                    getattr(tree_a.tree_, children)[node_a],
                    getattr(tree_b.tree_, children)[node_b],
                    depth + 1,
                    f"{path} > {side}",
                ))

    return differences


def trees_identical(tree_a, tree_b, tol=1e-9):
    """
    Check whether two fitted trees have the same structure.

    Identical means the same shape, the same feature and threshold at every
    split, and the same predicted class at every leaf. Impurity values are
    ignored, since Gini and Entropy never agree on those.

    Args:
        tree_a, tree_b: Fitted DecisionTreeClassifiers
        tol: Largest threshold gap still treated as equal

    Returns:
        True if the trees are structurally identical
    """
    return not _diff(tree_a, tree_b, max_depth=None, feature_names=None, tol=tol)


def compare_within_depth(tree_a, tree_b, max_depth, feature_names=None, tol=1e-9):
    """
    Find where two trees differ in their top max_depth levels.

    The region is what a tree limited to max_depth would show: nodes at depth
    0 to max_depth - 1. Anything deeper is ignored. Below a split on different
    features the walk stops, so only the divergence itself is reported.

    Args:
        tree_a, tree_b: Fitted DecisionTreeClassifiers
        max_depth: Number of levels to compare, counted from the root
        feature_names: Optional list of feature names for readability
        tol: Largest threshold gap still treated as equal

    Returns:
        TreeComparison with one NodeDifference per differing node
    """
    return TreeComparison(
        max_depth=max_depth,
        differences=_diff(tree_a, tree_b, max_depth, feature_names, tol),
    )


def format_comparison(comparison, labels=("a", "b")):
    """
    Render a TreeComparison as a printable report.

    Args:
        comparison: TreeComparison from compare_within_depth
        labels: Names for the two trees, e.g. ("gini", "entropy")
    """
    if comparison.identical_within_depth:
        return f"Trees match within the top {comparison.max_depth} levels."

    lines = [
        f"Trees differ within the top {comparison.max_depth} levels "
        f"({len(comparison.differences)} found, "
        f"first at depth {comparison.first_divergence_depth}):"
    ]
    for d in comparison.differences:
        lines.append(f"  [depth {d.depth}] {d.path} ({d.kind})")
        lines.append(f"      {labels[0]}: {d.a}")
        lines.append(f"      {labels[1]}: {d.b}")
    return "\n".join(lines)
