import numpy as np
from sklearn.tree import DecisionTreeClassifier

from analysis.tree_comparison import (
    compare_within_depth,
    format_comparison,
    trees_identical,
)


def fit(X, y, **kwargs):
    return DecisionTreeClassifier(random_state=0, **kwargs).fit(np.array(X), np.array(y))


# One feature, split at 1.5
STEP_X = [[0], [1], [2], [3]]
STEP_Y = [0, 0, 1, 1]

# Two features; the label follows whichever column is passed as y
TWO_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
FOLLOWS_FIRST = [0, 0, 1, 1]
FOLLOWS_SECOND = [0, 1, 0, 1]

# One feature, alternating labels: needs more than two levels of splits
ALT_X = [[i] for i in range(8)]
ALT_Y = [0, 1, 0, 1, 0, 1, 0, 1]


def test_tree_is_identical_to_itself():
    tree = fit(STEP_X, STEP_Y)
    assert trees_identical(tree, tree)


def test_refitted_tree_is_identical():
    assert trees_identical(fit(STEP_X, STEP_Y), fit(STEP_X, STEP_Y))


def test_same_structure_under_different_criteria_is_identical():
    gini = fit(STEP_X, STEP_Y, criterion="gini")
    entropy = fit(STEP_X, STEP_Y, criterion="entropy")
    assert trees_identical(gini, entropy)


def test_threshold_difference_is_not_identical():
    a = fit(STEP_X, STEP_Y)
    b = fit([[0], [1], [3], [4]], STEP_Y)
    assert not trees_identical(a, b)


def test_threshold_difference_is_reported_at_root():
    a = fit(STEP_X, STEP_Y)
    b = fit([[0], [1], [3], [4]], STEP_Y)

    comparison = compare_within_depth(a, b, max_depth=3)

    assert not comparison.identical_within_depth
    assert comparison.first_divergence_depth == 0
    assert [(d.path, d.depth, d.kind) for d in comparison.differences] == [
        ("root", 0, "threshold")
    ]


def test_threshold_difference_within_tolerance_is_identical():
    a = fit(STEP_X, STEP_Y)
    b = fit([[0], [1], [2 + 1e-12], [3]], STEP_Y)
    assert trees_identical(a, b)


def test_feature_difference_is_reported_with_feature_names():
    a = fit(TWO_X, FOLLOWS_FIRST)
    b = fit(TWO_X, FOLLOWS_SECOND)

    comparison = compare_within_depth(a, b, max_depth=3, feature_names=["age", "chol"])

    assert not trees_identical(a, b)
    assert comparison.first_divergence_depth == 0
    [difference] = comparison.differences
    assert difference.kind == "feature"
    assert "age" in difference.a
    assert "chol" in difference.b


def test_difference_below_the_region_is_not_reported():
    shallow = fit(ALT_X, ALT_Y, max_depth=2)
    deep = fit(ALT_X, ALT_Y, max_depth=3)

    comparison = compare_within_depth(shallow, deep, max_depth=2)

    assert not trees_identical(shallow, deep)
    assert comparison.identical_within_depth
    assert comparison.first_divergence_depth is None
    assert comparison.differences == []


def test_leaf_against_split_is_a_shape_difference():
    shallow = fit(ALT_X, ALT_Y, max_depth=2)
    deep = fit(ALT_X, ALT_Y, max_depth=3)

    comparison = compare_within_depth(shallow, deep, max_depth=3)

    assert comparison.first_divergence_depth == 2
    assert {d.kind for d in comparison.differences} == {"shape"}


def test_leaves_predicting_different_classes_differ():
    a = fit(STEP_X, [0, 0, 0, 0])
    b = fit(STEP_X, [1, 1, 1, 1])

    comparison = compare_within_depth(a, b, max_depth=1)

    assert not trees_identical(a, b)
    assert [d.kind for d in comparison.differences] == ["leaf_class"]


def test_format_reports_match_within_region():
    tree = fit(STEP_X, STEP_Y)
    report = format_comparison(compare_within_depth(tree, tree, max_depth=3))
    assert "match" in report
    assert "3" in report


def test_format_lists_each_difference_with_labels():
    a = fit(TWO_X, FOLLOWS_FIRST)
    b = fit(TWO_X, FOLLOWS_SECOND)
    comparison = compare_within_depth(a, b, max_depth=3, feature_names=["age", "chol"])

    report = format_comparison(comparison, labels=("gini", "entropy"))

    assert "root" in report
    assert "gini: age" in report
    assert "entropy: chol" in report
