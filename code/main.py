"""
Main orchestration script for XAI Gini vs. Entropy research.

Flow:
  1. Load config (paths, random seed, dataset names)
  2. Download datasets (cached in code/data/raw)
  3. Preprocess each dataset
  4. Train Gini and Entropy trees on each dataset
  5. Save trained models, tree diagrams and tree rules
  6. Write results/summary.csv

TODO: After training, run analysis modules for interpretability metrics, see idea-proposal.md for potential analysis dimensions.
"""

import joblib
import pandas as pd

from config import DATASETS, OUTPUT_DIR, RANDOM_STATE
from data.download import download_datasets
from data.preprocess import preprocess_dataset
from models.train import save_tree, train_entropy_tree, train_gini_tree

TRAINERS = {"gini": train_gini_tree, "entropy": train_entropy_tree}


def main():
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset_name in DATASETS:
        print(dataset_name)
        raw_data = download_datasets(dataset_name)
        X_train, X_test, y_train, y_test, feature_names = preprocess_dataset(
            raw_data, dataset_name
        )
        for criterion, train in TRAINERS.items():
            tree, accuracy = train(X_train, X_test, y_train, y_test,
                                   random_state=RANDOM_STATE)
            name = f"{dataset_name}_{criterion}"
            joblib.dump(tree, models_dir / f"{name}.pkl")
            save_tree(tree, feature_names, OUTPUT_DIR, name)
            rows.append({
                "dataset": dataset_name,
                "criterion": criterion,
                "test_accuracy": accuracy,
                "depth": tree.get_depth(),
                "n_nodes": tree.tree_.node_count,
                "n_leaves": tree.get_n_leaves(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
