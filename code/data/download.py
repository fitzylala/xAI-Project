import pandas as pd
from ucimlrepo import fetch_ucirepo

from config import RAW_DATA_DIR

UCI_IDS = {
    "breast_cancer_wisconsin": 17,
    "heart_disease": 45,
    "heart_failure": 519,
}


def download_datasets(dataset_name):
    if dataset_name not in UCI_IDS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; expected one of {sorted(UCI_IDS)}"
        )
    path = RAW_DATA_DIR / f"{dataset_name}.csv"
    if path.exists():
        return pd.read_csv(path)
    data = fetch_ucirepo(id=UCI_IDS[dataset_name]).data
    raw = data.features.assign(target=data.targets.iloc[:, 0])
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    return raw
