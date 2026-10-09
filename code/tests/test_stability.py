import pandas as pd
import pytest
from sklearn.datasets import make_classification

from analysis.stability import feature_importance_runs, stability_summary

COLUMNS = ["a", "b", "c", "d"]


def test_identical_runs_are_fully_stable():
    importances = pd.DataFrame([[0.5, 0.3, 0.2, 0.0]] * 3, columns=COLUMNS)

    summary = stability_summary(importances, top_k=2)

    assert summary["mean_kendall_tau"] == pytest.approx(1.0)
    assert summary["mean_top_k_jaccard"] == pytest.approx(1.0)


def test_disjoint_top_features_give_zero_jaccard():
    importances = pd.DataFrame(
        [[0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]], columns=COLUMNS
    )

    assert stability_summary(importances, top_k=2)["mean_top_k_jaccard"] == 0.0


def test_jaccard_is_averaged_against_the_first_run():
    importances = pd.DataFrame(
        [[0.6, 0.4, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.0, 0.0, 0.6, 0.4]],
        columns=COLUMNS,
    )

    assert stability_summary(importances, top_k=2)["mean_top_k_jaccard"] == pytest.approx(0.5)


def test_reversed_ranking_gives_negative_tau():
    importances = pd.DataFrame(
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], columns=COLUMNS
    )

    assert stability_summary(importances, top_k=2)["mean_kendall_tau"] == pytest.approx(-1.0)


def test_single_run_is_rejected():
    importances = pd.DataFrame([[0.5, 0.3, 0.2, 0.0]], columns=COLUMNS)

    with pytest.raises(ValueError):
        stability_summary(importances, top_k=2)


def _data():
    X, y = make_classification(n_samples=120, n_features=4, random_state=0)
    return pd.DataFrame(X, columns=COLUMNS), pd.Series(y)


def test_importance_runs_has_one_row_per_run_and_one_column_per_feature():
    X, y = _data()

    importances = feature_importance_runs(X, y, criterion="gini", n_runs=5)

    assert importances.shape == (5, 4)
    assert list(importances.columns) == COLUMNS


def test_importance_runs_resample_the_training_data():
    X, y = _data()

    importances = feature_importance_runs(X, y, criterion="entropy", n_runs=5)

    assert not importances.iloc[0].equals(importances.iloc[1])


def test_importance_runs_are_reproducible():
    X, y = _data()

    first = feature_importance_runs(X, y, criterion="gini", n_runs=3)
    second = feature_importance_runs(X, y, criterion="gini", n_runs=3)

    pd.testing.assert_frame_equal(first, second)
