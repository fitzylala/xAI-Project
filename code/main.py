"""
Main orchestration script for XAI Gini vs. Entropy research.

Flow:
  1. Load config (paths, random seed, dataset names)
  2. Download datasets
  3. Preprocess each dataset
  4. Train Gini and Entropy trees on each dataset
  5. Save trained models
  6. Print tree summaries (optional for inspection)

TODO: After training, run analysis modules for interpretability metrics, see idea-proposal.md for potential analysis dimensions.
"""

from config import RANDOM_STATE, DATASETS, OUTPUT_DIR
from data.download import download_datasets # TODO
from data.preprocess import preprocess_dataset # TODO
from models.train import train_gini_tree, train_entropy_tree, print_tree
import joblib


def setup():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    setup()
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    results = {}

    for dataset_name, uci_id in DATASETS.items():
        print(f"  {dataset_name}")

        print(f"    [1/5] Downloading...")
        raw_data = download_datasets(uci_id)
        print(f"    [2/5] Preprocessing...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_dataset(
            raw_data, dataset_name
        )

        print(f"    [3/5] Training Gini...")
        tree_gini, acc_gini = train_gini_tree(X_train, X_test, y_train, y_test,
                                              random_state=RANDOM_STATE)
        print(f"    [4/5] Training Entropy...")
        tree_entropy, acc_entropy = train_entropy_tree(X_train, X_test, y_train, y_test,
                                                       random_state=RANDOM_STATE)

        print(f"    [5/5] Saving models...")
        gini_path = models_dir / f"{dataset_name}_gini.pkl"
        entropy_path = models_dir / f"{dataset_name}_entropy.pkl"
        joblib.dump(tree_gini, gini_path)
        joblib.dump(tree_entropy, entropy_path)

        results[dataset_name] = {
            "gini": {"model": tree_gini, "accuracy": acc_gini},
            "entropy": {"model": tree_entropy, "accuracy": acc_entropy},
            "feature_names": feature_names,
        }

        print(f"    Gini: {acc_gini:.4f} | Entropy: {acc_entropy:.4f}")

    # Optional: inspect trees visually
    print("\nGenerating tree visualizations...")
    for dataset_name, data in results.items():
        print(f"  {dataset_name}")
        print(f"    Gini tree:")
        print_tree(data["gini"]["model"], feature_names=data["feature_names"])

        print(f"    Entropy tree:")
        print_tree(data["entropy"]["model"], feature_names=data["feature_names"])

    # TODO: Run analysis modules to compare interpretability metrics
    print("Done!")


if __name__ == "__main__":
    main()
