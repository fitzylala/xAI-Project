# xAI-Project

Group project for the Explainable AI module (University of Limerick).

## Topic

**Does the split criterion (Gini impurity vs. entropy) change how interpretable a decision tree is on medical data?**

We train matched Gini and entropy trees (scikit-learn `DecisionTreeClassifier`) on three UCI medical datasets and compare them using four interpretability proxies:

1. **Structure:** depth, leaves, nodes, features used, mean path length.
2. **Accuracy at fixed depth budgets:** max depth 3, 5, 7 and 10.
3. **Stability:** Kendall's tau and top-5 Jaccard overlap of feature-importance rankings across 30 resampled train/test splits.
4. **Readable levels:** node-by-node comparison of the top 3 levels of each tree.

Hypotheses (from [`project-work/skye/idea-proposal.md`](project-work/skye/idea-proposal.md)): H0 no systematic difference; H1 Gini gives simpler but less stable trees; H2 entropy gives more stable but deeper trees; H3 the difference depends on the dataset.

### Datasets

| Dataset | UCI id | Notes |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | 17 | Malignant = 1 |
| Heart Disease (Cleveland) | 45 | `num > 0` = 1; `ca`/`thal` imputed; nominal columns one-hot encoded |
| Heart Failure Clinical Records | 519 | `time` dropped (leaks the outcome) |

Why these datasets: [`project-work/aaron/research/datasets.md`](project-work/aaron/research/datasets.md).

## Structure

- [`code/`](code/) — the experiment pipeline (see below).
- [`paper/`](paper/) — the report (IEEE format, Org mode source in `paper.org`, references in `references.bib`). Build instructions in [`paper/README.md`](paper/README.md). The report is still in draft.
- [`journal/`](journal/) — dated log of meetings, decisions and reasoning (one file per entry, e.g. `journal/2026-09-21.md`).
- [`project-work/`](project-work/) — individual research notes, proposals and exploratory notebooks (`aaron/`, `skye/`).
- [`Presentation-Final-Submission.pptx`](Presentation-Final-Submission.pptx) — the submitted presentation.
- [`SPEC.md`](SPEC.md) — assignment specification for this journal.

## Running the code

The code targets Python 3 with the pinned versions in `code/requirements.txt`. Run everything from `code/`:

```sh
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py     # download, preprocess, train, compare, write results
pytest tests       # unit tests
```

`main.py` downloads the datasets from UCI, trains Gini and entropy trees on a stratified 80/20 split (`random_state=42`), saves the models and prints the depth-3 comparison. For datasets whose trees are not identical, it also computes the interpretability metrics and saves them to `results/interpretability_metrics.csv` (at the repo root). Experiment settings (seed, split, depth budgets, stability runs) are in [`code/config.py`](code/config.py).

| Path | Contents |
|---|---|
| `code/data/` | `download.py` (UCI fetch), `preprocess.py` (cleaning and encoding) |
| `code/models/train.py` | train/test split, Gini and entropy training, tree printing |
| `code/analysis/structural.py` | structural metrics and accuracy by depth |
| `code/analysis/stability.py` | feature-importance stability across resamples |
| `code/analysis/tree_comparison.py` | node-by-node comparison of two trees |
| `code/analysis/disease-analysis.py` | classes ported from earlier notebooks (data pipeline, Gini vs. entropy, SHAP/LIME diagnosis) |
| `project-work/aaron/notebooks/` | the original notebooks the results were first recorded in |

## Team

Aaron McGuinness, Mark James Drohan, Skye Fitzpatrick, Mikey Fennelly.
