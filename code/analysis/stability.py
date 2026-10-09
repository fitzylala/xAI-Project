"""
Stability of a tree's feature importances across resampled training sets.

If the important features change whenever the training rows change, the tree's
explanation can't be trusted. See project-work/skye/idea-proposal.md, Phase 3.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def feature_importance_runs(X, y, criterion, n_runs=30, test_size=0.2, stratify=True,
                            random_state=42):
    """
    Train one tree per random train/test split and collect feature importances.

    The split seed is the run number, so only the training rows change between
    runs; the tree's own seed stays fixed.

    Args:
        X: Feature DataFrame (preprocessed, unsplit)
        y: Target labels
        criterion: 'gini' or 'entropy'
        n_runs: Number of resampled splits
        test_size: Fraction of the data held out in each split
        stratify: Preserve the class balance in each split
        random_state: Random seed for the tree

    Returns:
        DataFrame of importances, one row per run and one column per feature
    """
    rows = []
    for run in range(n_runs):
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=run,
            stratify=y if stratify else None,
        )
        tree = DecisionTreeClassifier(criterion=criterion, random_state=random_state)
        tree.fit(X_train, y_train)
        rows.append(tree.feature_importances_)
    return pd.DataFrame(rows, columns=X.columns)


def stability_summary(importances, top_k=5):
    """
    Summarise how consistent feature importance rankings are across runs.

    Every run is compared with the first run (not all pairs), matching
    GiniVsEntropyExplainability in disease-analysis.py.

    Args:
        importances: DataFrame from feature_importance_runs
        top_k: Number of top features compared between runs

    Returns:
        dict with mean_kendall_tau (rank correlation of all features) and
        mean_top_k_jaccard (overlap of the top_k features)
    """
    if len(importances) < 2:
        raise ValueError("Stability requires at least two runs")

    reference = importances.iloc[0]
    reference_top = set(reference.nlargest(top_k).index)
    taus, jaccards = [], []
    for _, row in importances.iloc[1:].iterrows():
        tau, _ = kendalltau(reference.rank(), row.rank())
        taus.append(tau)
        top = set(row.nlargest(top_k).index)
        jaccards.append(len(reference_top & top) / len(reference_top | top))
    return {
        "mean_kendall_tau": float(np.mean(taus)),
        "mean_top_k_jaccard": float(np.mean(jaccards)),
    }
