from sklearn.datasets import make_classification

from models.train import save_tree, train_gini_tree


def test_save_tree_writes_png_and_rules(tmp_path):
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    tree, _ = train_gini_tree(X[:40], X[40:], y[:40], y[40:], random_state=0)
    save_tree(tree, ["f0", "f1", "f2", "f3"], tmp_path, "demo_gini")
    assert (tmp_path / "trees" / "demo_gini.png").stat().st_size > 0
    assert "f" in (tmp_path / "rules" / "demo_gini.txt").read_text()
