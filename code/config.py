from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE = 42

# Test/train split
TEST_SIZE = 0.2
STRATIFY = True

# Top levels of a tree that a human would read; trees are compared within these
EXPLAINABILITY_DEPTH = 3

# Depth limits for the accuracy-interpretability tradeoff
DEPTH_BUDGETS = (3, 5, 7, 10)

# Stability: number of resampled splits, and how many top features to compare
STABILITY_RUNS = 30
STABILITY_TOP_K = 5

# Datasets to analyze (from Phase 2 plan), mapped to their UCI repository ids
DATASETS = {
    "breast_cancer_wisconsin": 17,  # https://doi.org/10.24432/C5DW2B
    "heart_disease": 45,            # https://doi.org/10.24432/C52P4X
    "heart_failure": 519,           # https://doi.org/10.24432/C5Z89R
}

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results"
PROCESSED_DATA_DIR = PROJECT_ROOT / "code" / "data" / "processed"
