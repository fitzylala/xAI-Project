"""
Main orchestration script for XAI Gini vs. Entropy research.

Flow:
  1. Load config (paths, random seed, dataset names)
  2. Download datasets
  3. Preprocess each dataset
  4. Train Gini and Entropy trees on each dataset
  5. Save trained models
  6. Print tree summaries (optional for inspection)
  7. Check whether the two trees are identical; if not, report where they
     differ within the top EXPLAINABILITY_DEPTH levels
  8. For trees that differ, compute interpretability metrics (structural
     complexity, accuracy at fixed depths, stability) and save them as a CSV
"""

from config import (
    RANDOM_STATE, TEST_SIZE, STRATIFY, DATASETS, OUTPUT_DIR, EXPLAINABILITY_DEPTH,
    DEPTH_BUDGETS, STABILITY_RUNS, STABILITY_TOP_K,
)
from analysis.tree_comparison import (
    trees_identical, compare_within_depth, format_comparison,
)
from analysis.structural import structural_metrics, accuracy_by_depth
from analysis.stability import feature_importance_runs, stability_summary
from data.download import download_datasets
from data.preprocess import preprocess_dataset
from models.train import split_data, train_gini_tree, train_entropy_tree, print_tree
import joblib
import pandas as pd


def setup():
    """Create output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def interpretability_metrics(data, criterion):
    """Structural, accuracy-by-depth and stability metrics for one criterion."""
    X_train, X_test, y_train, y_test = data["split"]
    by_depth = accuracy_by_depth(
        X_train, X_test, y_train, y_test, criterion=criterion,
        depths=DEPTH_BUDGETS, random_state=RANDOM_STATE,
    )
    importances = feature_importance_runs(
        data["X"], data["y"], criterion=criterion, n_runs=STABILITY_RUNS,
        test_size=TEST_SIZE, stratify=STRATIFY, random_state=RANDOM_STATE,
    )
    return {
        "accuracy": data[criterion]["accuracy"],
        **structural_metrics(data[criterion]["model"]),
        **{f"accuracy_depth_{depth}": acc for depth, acc in by_depth.items()},
        **stability_summary(importances, top_k=STABILITY_TOP_K),
    }


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
        X, y = preprocess_dataset(raw_data, dataset_name)
        feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=TEST_SIZE, stratify=STRATIFY, random_state=RANDOM_STATE
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
            "X": X,
            "y": y,
            "split": (X_train, X_test, y_train, y_test),
            "identical": trees_identical(tree_gini, tree_entropy),
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

    print("\nComparing Gini and Entropy trees...")
    metric_rows = []
    for dataset_name, data in results.items():
        print(f"  {dataset_name}")
        if data["identical"]:
            print("    Trees are identical, skipping comparison.")
            continue

        comparison = compare_within_depth(
            data["gini"]["model"], data["entropy"]["model"],
            max_depth=EXPLAINABILITY_DEPTH, feature_names=data["feature_names"],
        )
        report = format_comparison(comparison, labels=("gini", "entropy"))
        print("    " + report.replace("\n", "\n    "))

        for criterion in ("gini", "entropy"):
            metric_rows.append({
                "dataset": dataset_name, "criterion": criterion,
                **interpretability_metrics(data, criterion),
            })

    if metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics_path = OUTPUT_DIR / "interpretability_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        print("\nInterpretability metrics:")
        print(metrics.round(3).set_index(["dataset", "criterion"]).T.to_string())
        print(f"\nSaved to {metrics_path}")

    print("Done!")


if __name__ == "__main__":
    main()
