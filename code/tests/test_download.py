from types import SimpleNamespace

import pandas as pd
import pytest

from data import download


def fake_fetch(id):
    features = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    targets = pd.DataFrame({"label": [0, 1]})
    return SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))


def failing_fetch(id):
    raise ConnectionError("offline")


def test_unknown_dataset_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="heart_disease"):
        download.download_datasets("nope")


def test_fetch_writes_cache_with_target_column(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "fetch_ucirepo", fake_fetch)
    raw = download.download_datasets("heart_failure")
    assert list(raw.columns) == ["a", "b", "target"]
    assert (tmp_path / "heart_failure.csv").exists()


def test_cached_csv_is_used_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    pd.DataFrame({"a": [9], "target": [1]}).to_csv(tmp_path / "heart_failure.csv", index=False)
    monkeypatch.setattr(download, "fetch_ucirepo", failing_fetch)
    raw = download.download_datasets("heart_failure")
    assert raw["a"].tolist() == [9]


def test_failed_fetch_leaves_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(download, "fetch_ucirepo", failing_fetch)
    with pytest.raises(ConnectionError):
        download.download_datasets("heart_failure")
    assert not (tmp_path / "heart_failure.csv").exists()
