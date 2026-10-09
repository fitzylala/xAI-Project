# clean & encode data
import pandas as pd

# Nominal heart disease features stored as integers in the raw data
NOMINAL_COLS_HD = ["cp", "restecg", "slope", "thal"]

# `time` is the follow-up period, which leaks the outcome
EXCLUDE_TIME_HF = True


def _preprocess_breast_cancer(X, y):
    y = (y == "M").astype(int)  # 1 = malignant
    return X, y


def _preprocess_heart_disease(X, y):
    y = (y > 0).astype(int)  # 1 = any disease

    X["ca"] = X["ca"].fillna(X["ca"].median())
    X["thal"] = X["thal"].fillna(X["thal"].mode().iloc[0])

    X = pd.get_dummies(X, columns=NOMINAL_COLS_HD, prefix=NOMINAL_COLS_HD)
    X = X.astype({c: int for c in X.columns if X[c].dtype == bool})
    return X, y


def _preprocess_heart_failure(X, y):
    if EXCLUDE_TIME_HF:
        X = X.drop(columns=["time"])
    return X, y


_PREPROCESSORS = {
    "breast_cancer_wisconsin": _preprocess_breast_cancer,
    "heart_disease": _preprocess_heart_disease,
    "heart_failure": _preprocess_heart_failure,
}


def preprocess_dataset(raw_data, dataset_name):
    """Clean and encode a raw (X, y) pair from download_datasets.

    Returns the cleaned (X, y), unsplit.
    """
    if dataset_name not in _PREPROCESSORS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Expected one of {list(_PREPROCESSORS)}"
        )

    X, y = raw_data
    return _PREPROCESSORS[dataset_name](X.copy(), y.copy())
