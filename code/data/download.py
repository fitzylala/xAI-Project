# download the datasets
from ucimlrepo import fetch_ucirepo


def download_datasets(uci_id):
    """Fetch a raw dataset from the UCI repository by its id.

    Returns (X, y): the untouched features DataFrame and target Series.
    Cleaning, encoding and target construction are left to preprocessing.
    """
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features.copy()
    y = ds.data.targets.copy().iloc[:, 0]
    return X, y
