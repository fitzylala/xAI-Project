from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def train_entropy_tree(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train a decision tree using entropy (information gain) as splitting criterion.

    Args:
        X_train, X_test: Feature arrays (preprocessed)
        y_train, y_test: Target labels
        random_state: Random seed for reproducibility

    Returns:
        tree: Fitted DecisionTreeClassifier with criterion='entropy'
        test_accuracy: Float, accuracy on test set
    """
    tree = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
    tree.fit(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)
    return tree, test_accuracy


def train_gini_tree(X_train, X_test, y_train, y_test, random_state=42):
    """
    Train a decision tree using Gini impurity as splitting criterion.

    Args:
        X_train, X_test: Feature arrays (preprocessed)
        y_train, y_test: Target labels
        random_state: Random seed for reproducibility

    Returns:
        tree: Fitted DecisionTreeClassifier with criterion='gini'
        test_accuracy: Float, accuracy on test set
    """
    tree = DecisionTreeClassifier(criterion='gini', random_state=random_state)
    tree.fit(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)
    return tree, test_accuracy


def save_tree(tree, feature_names, out_dir, name, figsize=(20, 10)):
    rules_dir = out_dir / "rules"
    trees_dir = out_dir / "trees"
    rules_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / f"{name}.txt").write_text(
        export_text(tree, feature_names=feature_names)
    )
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True,
              fontsize=9, ax=fig.subplots())
    fig.savefig(trees_dir / f"{name}.png", dpi=150, bbox_inches="tight")
