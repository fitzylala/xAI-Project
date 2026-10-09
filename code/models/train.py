import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def split_data(X, y, test_size=0.2, stratify=True, random_state=42):
    """
    Split preprocessed data into train and test sets.

    Args:
        X: Feature DataFrame (preprocessed)
        y: Target labels
        test_size: Fraction of the data held out for testing
        stratify: Preserve the class balance in both splits
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


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


def print_tree(tree, feature_names=None, class_names=None, figsize=(20, 10)):
    """
    Print tree structure and basic stats, then display visual diagram.

    Args:
        tree: Fitted DecisionTreeClassifier
        feature_names: Optional list of feature names for readability
        class_names: Optional list of class names (e.g., ["negative", "positive"])
        figsize: Tuple (width, height) for the figure size
    """
    print(f"\nTree Depth:  {tree.get_depth()}")
    print(f"N Leaves:    {tree.get_n_leaves()}")
    print(f"N Nodes:     {tree.tree_.node_count}")

    print("\nTree Rules:")
    rules = export_text(tree, feature_names=feature_names)
    print(rules)

    plt.figure(figsize=figsize)
    plot_tree(tree, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=9)
    plt.tight_layout()
    plt.show()
