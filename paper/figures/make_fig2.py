"""Regenerate Fig. 2 (stability of feature-importance rankings) from the repo code.

Run from the repo root:  python paper/figures/make_fig2.py

Each run trains an unconstrained tree on one stratified 80/20 split (split seed = run
number, tree seed fixed) via analysis.stability.feature_importance_runs. Every pair of
the 30 runs is compared (435 pairs), which does not depend on which run is first.
The whiskers show the range of that all-pairs mean over 5 disjoint blocks of 30 split
seeds, i.e. how much the number moves if a different 30 resamples are drawn.
"""
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from config import DATASETS, RANDOM_STATE, STABILITY_RUNS, STABILITY_TOP_K, STRATIFY, TEST_SIZE
from data.download import download_datasets
from data.preprocess import preprocess_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BLOCKS = 5
LABELS = {"breast_cancer_wisconsin": "BC", "heart_disease": "HD", "heart_failure": "HF"}
COLORS = {"gini": "#2a78d6", "entropy": "#eb6834"}
HATCH = {"gini": "", "entropy": "///"}


def importances(X, y, criterion, seeds):
    rows = []
    for seed in seeds:
        X_tr, _, y_tr, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y if STRATIFY else None)
        tree = DecisionTreeClassifier(criterion=criterion, random_state=RANDOM_STATE).fit(X_tr, y_tr)
        rows.append(tree.feature_importances_)
    return pd.DataFrame(rows, columns=X.columns)


def all_pairs(imp, top_k=STABILITY_TOP_K):
    ranks = [imp.iloc[i].rank() for i in range(len(imp))]
    tops = [set(imp.iloc[i].nlargest(top_k).index) for i in range(len(imp))]
    taus, jacs = [], []
    for i, j in combinations(range(len(imp)), 2):
        taus.append(kendalltau(ranks[i], ranks[j])[0])
        jacs.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    return float(np.mean(taus)), float(np.mean(jacs))


rows = []
for name, uci_id in DATASETS.items():
    X, y = preprocess_dataset(download_datasets(uci_id), name)
    for criterion in ("gini", "entropy"):
        blocks = [all_pairs(importances(X, y, criterion,
                  range(b * STABILITY_RUNS, (b + 1) * STABILITY_RUNS))) for b in range(BLOCKS)]
        tau, jac = zip(*blocks)
        rows.append({"dataset": LABELS[name], "criterion": criterion,
                     "tau": tau[0], "tau_min": min(tau), "tau_max": max(tau),
                     "jaccard": jac[0], "jaccard_min": min(jac), "jaccard_max": max(jac)})
df = pd.DataFrame(rows)
df.round(3).to_csv(Path(__file__).with_name("fig2_data.csv"), index=False)
print(df.round(3).to_string(index=False))

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=True)
width = 0.38
for ax, key, title in zip(axes, ("tau", "jaccard"), ("Kendall's $\\tau$", "Top-5 Jaccard")):
    for k, criterion in enumerate(("gini", "entropy")):
        sub = df[df.criterion == criterion].set_index("dataset").loc[list(LABELS.values())]
        x = np.arange(len(sub)) + (k - 0.5) * width
        err = [sub[key] - sub[f"{key}_min"], sub[f"{key}_max"] - sub[key]]
        ax.bar(x, sub[key], width, color=COLORS[criterion], hatch=HATCH[criterion],
               edgecolor="white", linewidth=0.6, label=criterion.capitalize())
        ax.errorbar(x, sub[key], yerr=err, fmt="none", ecolor="#333", elinewidth=0.8, capsize=1.5)
    ax.set_xticks(range(3), list(LABELS.values()))
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#ddd", linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Mean agreement, all run pairs")
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("fig2_stability.pdf"))
fig.savefig(Path(__file__).with_name("fig2_stability.png"), dpi=300)
