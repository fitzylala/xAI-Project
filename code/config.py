from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE = 42

# Test/train split
TEST_SIZE = 0.2
STRATIFY = True

# Datasets to analyze (from Phase 2 plan)
DATASETS = [
    "breast_cancer_wisconsin",
    "heart_disease",
    "heart_failure",
]

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results"
PROCESSED_DATA_DIR = PROJECT_ROOT / "code" / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "code" / "data" / "raw"
