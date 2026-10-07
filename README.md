# xAI-Project

Group project in Explainable AI (XAI).

## Topic

Does the choice of splitting criterion, Gini impurity or entropy (information gain), change how interpretable a decision tree is?

We train both kinds of tree on three medical datasets from the UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic, Heart Disease, Heart Failure Clinical Records) and compare them. The proposal is in [`project-work/skye/idea-proposal.md`](project-work/skye/idea-proposal.md) and the dataset reasoning in [`project-work/aaron/research/datasets.md`](project-work/aaron/research/datasets.md).

## Running the pipeline

    cd code
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python main.py

The first run downloads the datasets and saves them in `code/data/raw/`; later runs read those files and need no network. To run the tests, run `.venv/bin/python -m pytest tests`.

## What the pipeline writes

- `results/summary.csv`: test accuracy, depth, node count and leaf count for each dataset and criterion
- `results/models/`: the trained trees
- `results/trees/`: a diagram of each tree
- `results/rules/`: each tree as plain-text rules
- `code/data/processed/`: the cleaned train and test tables

All of these are committed, along with the raw downloads in `code/data/raw/`; `main.py` regenerates them.

## Structure

- [`journal/`](journal/): dated log of meetings, decisions and reasoning, one file per entry
- [`code/`](code/): the training pipeline
- [`project-work/`](project-work/): each member's research notes and notebooks
- [`paper/`](paper/): the report draft
- [`SPEC.md`](SPEC.md): the assignment specification for the journal

## Team

Skye Fitzpatrick, Mark James Drohan, Aaron McGuinness, Mikey Fennelly.
