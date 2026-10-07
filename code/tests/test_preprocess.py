import numpy as np
import pandas as pd
import pytest

from data import preprocess


@pytest.fixture(autouse=True)
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", tmp_path)
    return tmp_path


def make_heart_disease(n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(90, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n),
        "ca": np.arange(n, dtype=float) ** 2,
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": np.tile([0, 1, 2, 3], n // 4),
    })
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df


def test_heart_disease_has_no_missing_values_and_binary_target():
    X_train, X_test, y_train, y_test, _ = preprocess.preprocess_dataset(
        make_heart_disease(), "heart_disease"
    )
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert set(y_train) | set(y_test) == {0, 1}


def test_split_sizes_follow_config():
    X_train, X_test, y_train, y_test, _ = preprocess.preprocess_dataset(
        make_heart_disease(), "heart_disease"
    )
    assert (len(X_train), len(X_test)) == (32, 8)
    assert (len(y_train), len(y_test)) == (32, 8)


def test_ca_is_filled_from_training_rows_only():
    raw = make_heart_disease()
    X_train, X_test, _, _, _ = preprocess.preprocess_dataset(raw, "heart_disease")
    train_median = raw.loc[X_train.index, "ca"].median()
    assert train_median != raw["ca"].median()
    filled = pd.concat([X_train, X_test]).loc[[3, 17], "ca"]
    assert (filled == train_median).all()


def test_feature_names_match_frames_and_saved_csv(processed_dir):
    X_train, X_test, _, _, names = preprocess.preprocess_dataset(
        make_heart_disease(), "heart_disease"
    )
    assert names == list(X_train.columns) == list(X_test.columns)
    saved = pd.read_csv(processed_dir / "heart_disease_train.csv")
    assert list(saved.columns) == names + ["target"]
    assert (processed_dir / "heart_disease_test.csv").exists()
    assert not {"cp", "restecg", "slope", "thal"} & set(names)


def test_unseen_nominal_value_in_test_rows_is_encoded_as_zeros():
    raw = make_heart_disease().drop(columns="target")
    train = raw.iloc[:30]
    test = raw.iloc[30:].assign(cp=99)
    transformer = preprocess.heart_disease_transformer()
    transformer.fit(train)
    encoded = transformer.transform(test)
    cp_columns = [c for c in encoded.columns if c.startswith("cp_")]
    assert cp_columns
    assert (encoded[cp_columns] == 0).all().all()


def test_breast_cancer_maps_malignant_to_one():
    raw = pd.DataFrame({
        "radius1": np.linspace(5, 25, 20),
        "texture1": np.linspace(10, 30, 20),
        "target": ["M", "B"] * 10,
    })
    X_train, X_test, y_train, y_test, names = preprocess.preprocess_dataset(
        raw, "breast_cancer_wisconsin"
    )
    y = pd.concat([y_train, y_test]).sort_index()
    assert (y == (raw["target"] == "M").astype(int)).all()
    assert names == ["radius1", "texture1"]


def test_heart_failure_drops_time():
    raw = pd.DataFrame({
        "age": np.arange(20) + 40,
        "ejection_fraction": np.arange(20) + 20,
        "time": np.arange(20) * 10,
        "target": [0, 1] * 10,
    })
    _, _, _, _, names = preprocess.preprocess_dataset(raw, "heart_failure")
    assert names == ["age", "ejection_fraction"]


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="nope"):
        preprocess.preprocess_dataset(make_heart_disease(), "nope")
