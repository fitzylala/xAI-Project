from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from config import PROCESSED_DATA_DIR, RANDOM_STATE, STRATIFY, TEST_SIZE

HEART_DISEASE_NOMINAL = ["cp", "restecg", "slope", "thal"]


def heart_disease_transformer():
    nominal = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [
            ("ca", SimpleImputer(strategy="median"), ["ca"]),
            ("nominal", nominal, HEART_DISEASE_NOMINAL),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def preprocess_dataset(raw_data, dataset_name):
    X = raw_data.drop(columns="target")
    y = raw_data["target"]
    if dataset_name == "breast_cancer_wisconsin":
        y = (y == "M").astype(int)
    elif dataset_name == "heart_disease":
        y = (y > 0).astype(int)
    elif dataset_name == "heart_failure":
        X = X.drop(columns="time")
    else:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if STRATIFY else None,
    )

    if dataset_name == "heart_disease":
        transformer = heart_disease_transformer()
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.assign(target=y_train).to_csv(
        PROCESSED_DATA_DIR / f"{dataset_name}_train.csv", index=False
    )
    X_test.assign(target=y_test).to_csv(
        PROCESSED_DATA_DIR / f"{dataset_name}_test.csv", index=False
    )
    return X_train, X_test, y_train, y_test, list(X_train.columns)
