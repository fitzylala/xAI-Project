"""The classes here are ported from previous commits by Aaron McGuinness:

7fc9a4d: DataPipeline, GiniVsEntropyExplainability
86030a3 and 96a568d: MedicalDiagnosis

Example usage:
    pipeline = DataPipeline(
        out_dir=config.PROCESSED_DATA_DIR,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=config.STRATIFY,
        datasets=config.DATASETS,
    ).run()

    comparison = GiniVsEntropyExplainability(
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
        stratify=config.STRATIFY,
    ).run(pipeline.splits)

    diagnosis = MedicalDiagnosis(
        random_state=config.RANDOM_STATE,
        test_size=config.TEST_SIZE,
        stratify=config.STRATIFY,
    ).run()

Config properties are supplied by the caller. run() methods return dataclasses;
plot_results() methods return figures without displaying or saving them.

Authors: Aaron McGuinness, Mikey Fennelly.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ucimlrepo import fetch_ucirepo

if TYPE_CHECKING:
    from lime.explanation import Explanation as LimeExplanation
    from lime.lime_tabular import LimeTabularExplainer
    from shap import Explanation as ShapExplanation


@dataclass
class DatasetInspection:
    """Raw data and the metadata/quality checks displayed in the notebook."""

    name: str
    n_instances: int
    n_features: int
    citation: str
    features: pd.DataFrame
    target: pd.Series
    feature_shape: tuple[int, int]
    missing_values_total: int
    missing_values_per_column: pd.Series
    target_counts: pd.Series
    variables: pd.DataFrame


@dataclass
class PreparedDataset:
    """Processed features plus the raw inspection and notebook previews."""

    name: str
    features: pd.DataFrame
    target: pd.Series
    inspection: DatasetInspection
    feature_shape: tuple[int, int]
    missing_values_total: int
    preview: pd.DataFrame
    description: pd.DataFrame
    target_correlations: pd.Series | None = None


@dataclass
class DatasetSummary:
    dataset: str
    n_instances: int
    n_features: int
    n_train: int
    n_test: int
    positive_rate: float


@dataclass
class DatasetSplit:
    summary: DatasetSummary
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_path: Path
    test_path: Path


@dataclass
class PipelineResult:
    datasets: dict[str, PreparedDataset]
    splits: dict[str, DatasetSplit]
    summaries: list[DatasetSummary]


class DataPipeline:
    """Load, inspect, preprocess, and save the three UCI datasets on demand.

    Heart-disease imputation is performed before splitting, matching the
    source notebook. The pipeline does not change pandas display settings.
    """

    NOMINAL_COLS_HD = ["cp", "restecg", "slope", "thal"]

    def __init__(
        self,
        out_dir: Path,
        *,
        test_size: float,
        random_state: int,
        stratify: bool,
        datasets: Sequence[str],
    ) -> None:
        self.out_dir = Path(out_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.datasets = tuple(datasets)

    def inspect_dataset(self, dataset_id: int, citation: str) -> DatasetInspection:
        """Fetch once and return the notebook's raw-data diagnostics."""
        dataset = fetch_ucirepo(id=dataset_id)
        features = dataset.data.features.copy()
        target = dataset.data.targets.iloc[:, 0].copy()
        missing = features.isna().sum()
        counts = target.value_counts()
        if dataset_id == 45:
            counts = counts.sort_index()
        return DatasetInspection(
            name=dataset.metadata.name,
            n_instances=int(dataset.metadata.num_instances),
            n_features=int(dataset.metadata.num_features),
            citation=citation,
            features=features,
            target=target,
            feature_shape=features.shape,
            missing_values_total=int(missing.sum()),
            missing_values_per_column=missing[missing > 0],
            target_counts=counts,
            variables=dataset.variables[["name", "role", "type", "description"]].copy(),
        )

    def _prepared_dataset(
        self,
        name: str,
        inspection: DatasetInspection,
        features: pd.DataFrame,
        target: pd.Series,
        target_correlations: pd.Series | None = None,
    ) -> PreparedDataset:
        return PreparedDataset(
            name=name,
            features=features,
            target=target,
            inspection=inspection,
            feature_shape=features.shape,
            missing_values_total=int(features.isna().sum().sum()),
            preview=features.head(),
            description=features.describe().T,
            target_correlations=target_correlations,
        )

    def load_breast_cancer(self) -> PreparedDataset:
        """Encode malignant diagnoses as 1 and benign diagnoses as 0."""
        raw = self.inspect_dataset(17, "https://doi.org/10.24432/C5DW2B")
        return self._prepared_dataset(
            "breast_cancer", raw, raw.features.copy(), (raw.target == "M").astype(int)
        )

    def load_heart_disease(self) -> PreparedDataset:
        """Impute missing values, encode nominal features, and binarize severity."""
        raw = self.inspect_dataset(45, "https://doi.org/10.24432/C52P4X")
        features = raw.features.copy()
        features["ca"] = features["ca"].fillna(features["ca"].median())
        features["thal"] = features["thal"].fillna(features["thal"].mode().iloc[0])
        features = pd.get_dummies(
            features, columns=self.NOMINAL_COLS_HD, prefix=self.NOMINAL_COLS_HD
        )
        features = features.astype({c: int for c in features if features[c].dtype == bool})
        return self._prepared_dataset(
            "heart_disease", raw, features, (raw.target > 0).astype(int)
        )

    def load_heart_failure(self, exclude_time: bool = True) -> PreparedDataset:
        """Return outcome correlations and optionally remove follow-up time."""
        raw = self.inspect_dataset(519, "https://doi.org/10.24432/C5Z89R")
        correlations = (
            raw.features.assign(death_event=raw.target)
            .corr(numeric_only=True)["death_event"]
            .drop("death_event")
            .sort_values(key=lambda values: values.abs(), ascending=False)
        )
        features = raw.features.copy()
        if exclude_time:
            features = features.drop(columns=["time"])
        return self._prepared_dataset(
            "heart_failure", raw, features, raw.target.copy(), correlations
        )

    def split_and_save(self, dataset: PreparedDataset) -> DatasetSplit:
        """Save a reproducible split and return its data, paths, and summary."""
        X, y = dataset.features, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.out_dir / f"{dataset.name}_train.csv"
        test_path = self.out_dir / f"{dataset.name}_test.csv"
        X_train.assign(target=y_train).to_csv(train_path, index=False)
        X_test.assign(target=y_test).to_csv(test_path, index=False)
        return DatasetSplit(
            summary=DatasetSummary(
                dataset=dataset.name,
                n_instances=len(X),
                n_features=X.shape[1],
                n_train=len(X_train),
                n_test=len(X_test),
                positive_rate=round(float(y.mean()), 3),
            ),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_path=train_path,
            test_path=test_path,
        )

    def run(self, exclude_time: bool = True) -> PipelineResult:
        """Process the configured datasets and return all notebook results."""
        loaders = {
            "breast_cancer_wisconsin": self.load_breast_cancer,
            "heart_disease": self.load_heart_disease,
            "heart_failure": lambda: self.load_heart_failure(exclude_time=exclude_time),
        }
        unknown = set(self.datasets) - loaders.keys()
        if unknown:
            raise ValueError(f"Unknown configured datasets: {sorted(unknown)}")
        datasets = {}
        splits = {}
        for name in self.datasets:
            dataset = loaders[name]()
            datasets[dataset.name] = dataset
            splits[dataset.name] = self.split_and_save(dataset)
        return PipelineResult(
            datasets=datasets,
            splits=splits,
            summaries=[split.summary for split in splits.values()],
        )

@dataclass
class TreeTrainingResult:
    tree: DecisionTreeClassifier
    accuracy: float


@dataclass
class StructuralMetrics:
    depth: int
    n_leaves: int
    n_nodes: int
    n_features_used: int


@dataclass
class StabilitySummary:
    mean_kendall_tau: float
    mean_top_k_jaccard: float


@dataclass
class StabilityResult:
    importances: pd.DataFrame
    summary: StabilitySummary


@dataclass
class FigureResult:
    figures: dict[str, Figure]


@dataclass
class GiniVsEntropyResult:
    baseline: dict[str, dict[str, TreeTrainingResult]]
    structural: pd.DataFrame
    accuracy_by_depth: pd.DataFrame
    stability: pd.DataFrame
    importances: dict[tuple[str, str], pd.DataFrame]
    combined: pd.DataFrame


class GiniVsEntropyExplainability:
    """Port of 02_gini_vs_entropy_explainability.ipynb at 7fc9a4d.

    Pass DataPipeline.run().splits to run(). Stability compares every resample
    with run zero, as in the notebook; it is not an all-pairs comparison.
    """

    CRITERIA = ("gini", "entropy")
    COLORS = {"gini": "#2a78d6", "entropy": "#eb6834"}

    def __init__(
        self, *, random_state: int, test_size: float, stratify: bool,
        depths: Sequence[int] = (3, 5, 7, 10), n_runs: int = 30,
        top_k: int = 5, viz_depth: int = 3,
    ) -> None:
        if n_runs < 2 or top_k < 1:
            raise ValueError("n_runs must be at least 2 and top_k must be positive")
        if not depths or any(depth < 1 for depth in depths) or viz_depth < 1:
            raise ValueError("Depth budgets must be positive")
        self.random_state = random_state
        self.test_size = test_size
        self.stratify = stratify
        self.depths = tuple(depths)
        self.n_runs = n_runs
        self.top_k = top_k
        self.viz_depth = viz_depth

    def train_tree(
        self, split: DatasetSplit, criterion: str, max_depth: int | None = None,
    ) -> TreeTrainingResult:
        tree = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth, random_state=self.random_state,
        )
        tree.fit(split.X_train, split.y_train)
        return TreeTrainingResult(tree, float(tree.score(split.X_test, split.y_test)))

    def structural_metrics(self, tree: DecisionTreeClassifier) -> StructuralMetrics:
        used = tree.tree_.feature
        return StructuralMetrics(
            tree.get_depth(), tree.get_n_leaves(), tree.tree_.node_count,
            len(set(used[used >= 0])),
        )

    def stability_summary(self, importances: pd.DataFrame) -> StabilitySummary:
        if len(importances) < 2 or importances.shape[1] == 0:
            raise ValueError("Stability requires at least two runs and one feature")
        reference = importances.iloc[0]
        reference_top = set(reference.nlargest(self.top_k).index)
        taus, jaccards = [], []
        for _, row in importances.iloc[1:].iterrows():
            tau, _ = kendalltau(reference.rank(), row.rank())
            taus.append(tau)
            top = set(row.nlargest(self.top_k).index)
            jaccards.append(len(reference_top & top) / len(reference_top | top))
        # Undefined tau (e.g. constant rankings) stays NaN, matching the notebook.
        return StabilitySummary(float(np.mean(taus)), float(np.mean(jaccards)))

    def run_stability_experiment(
        self, features: pd.DataFrame, target: pd.Series, criterion: str,
    ) -> StabilityResult:
        rows = []
        for run in range(self.n_runs):
            X_train, _, y_train, _ = train_test_split(
                features, target, test_size=self.test_size, random_state=run,
                stratify=target if self.stratify else None,
            )
            tree = DecisionTreeClassifier(criterion=criterion, random_state=self.random_state)
            tree.fit(X_train, y_train)
            rows.append(tree.feature_importances_)
        importances = pd.DataFrame(rows, columns=features.columns)
        return StabilityResult(importances, self.stability_summary(importances))

    def run(self, splits: Mapping[str, DatasetSplit]) -> GiniVsEntropyResult:
        if not splits:
            raise ValueError("At least one dataset split is required")
        baseline, importances = {}, {}
        structural_rows, depth_rows, stability_rows = [], [], []
        for name, split in splits.items():
            baseline[name] = {}
            features = pd.concat([split.X_train, split.X_test])
            target = pd.concat([split.y_train, split.y_test])
            for criterion in self.CRITERIA:
                trained = self.train_tree(split, criterion)
                baseline[name][criterion] = trained
                structural_rows.append({
                    "dataset": name, "criterion": criterion, "accuracy": trained.accuracy,
                    **asdict(self.structural_metrics(trained.tree)),
                })
                for depth in self.depths:
                    depth_rows.append({
                        "dataset": name, "criterion": criterion, "max_depth": depth,
                        "accuracy": self.train_tree(split, criterion, depth).accuracy,
                    })
                stability = self.run_stability_experiment(features, target, criterion)
                importances[name, criterion] = stability.importances
                stability_rows.append({
                    "dataset": name, "criterion": criterion, **asdict(stability.summary),
                })
        structural = pd.DataFrame(structural_rows)
        stability = pd.DataFrame(stability_rows)
        combined = structural.merge(stability, on=["dataset", "criterion"]).sort_values(
            ["dataset", "criterion"]
        ).reset_index(drop=True)
        return GiniVsEntropyResult(
            baseline, structural, pd.DataFrame(depth_rows), stability, importances, combined,
        )

    def plot_results(
        self, result: GiniVsEntropyResult, splits: Mapping[str, DatasetSplit],
    ) -> FigureResult:
        """Return structural, depth, stability, and depth-limited tree figures."""
        names = list(result.baseline)
        x, width = np.arange(len(names)), 0.35
        figures = {}
        for key, table, metrics in (
            ("structural", result.structural, ["depth", "n_leaves", "n_nodes", "n_features_used"]),
            ("stability", result.stability, ["mean_kendall_tau", "mean_top_k_jaccard"]),
        ):
            fig = Figure(figsize=(4 * len(metrics), 4))
            axes = fig.subplots(1, len(metrics), squeeze=False)[0]
            for ax, metric in zip(axes, metrics):
                for criterion, offset in zip(self.CRITERIA, (-width / 2, width / 2)):
                    values = table[table.criterion == criterion].set_index("dataset").loc[names, metric]
                    ax.bar(x + offset, values, width, label=criterion.title(), color=self.COLORS[criterion])
                ax.set_xticks(x, names, rotation=30, ha="right")
                ax.set_title(metric)
            axes[0].legend()
            fig.tight_layout()
            figures[key] = fig
        fig = Figure(figsize=(5 * len(names), 4))
        axes = fig.subplots(1, len(names), sharey=True, squeeze=False)[0]
        for ax, name in zip(axes, names):
            for criterion in self.CRITERIA:
                rows = result.accuracy_by_depth
                rows = rows[(rows.dataset == name) & (rows.criterion == criterion)].sort_values("max_depth")
                ax.plot(rows.max_depth, rows.accuracy, marker="o", label=criterion.title(),
                        color=self.COLORS[criterion])
            ax.set(title=name, xlabel="max_depth")
        axes[0].set_ylabel("test accuracy")
        axes[0].legend()
        fig.tight_layout()
        figures["accuracy_by_depth"] = fig
        for name in names:
            fig = Figure(figsize=(20, 6))
            FigureCanvasAgg(fig)
            for ax, criterion in zip(fig.subplots(1, 2), self.CRITERIA):
                trained = self.train_tree(splits[name], criterion, self.viz_depth)
                plot_tree(
                    trained.tree, feature_names=list(splits[name].X_train.columns),
                    class_names=["negative", "positive"], filled=True, rounded=True,
                    fontsize=8, ax=ax,
                )
                ax.set_title(f"{name} — {criterion} (depth={self.viz_depth}, acc={trained.accuracy:.3f})")
            fig.tight_layout()
            figures[f"trees_{name}"] = fig
        return FigureResult(figures)


@dataclass
class MedicalDataset:
    frame: pd.DataFrame
    shape: tuple[int, int]
    description: pd.DataFrame
    class_balance: pd.Series
    zero_counts: pd.Series
    correlations: pd.DataFrame


@dataclass
class MedicalSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    medians: pd.Series


@dataclass
class MedicalModel:
    model: RandomForestClassifier
    accuracy: float
    f1: float
    roc_auc: float
    classification_report: pd.DataFrame
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


@dataclass
class ShapResult:
    positive_class: "ShapExplanation"


@dataclass
class LimeResult:
    explainer: "LimeTabularExplainer"
    explanation: "LimeExplanation"
    patient_idx: int
    patient: pd.Series
    actual_outcome: int
    predicted_probability: float
    weights: list[tuple[str, float]]


@dataclass
class FeatureRanking:
    features: list[str]


@dataclass
class AgreementResult:
    patients: pd.DataFrame
    mean_jaccard: float
    feature_frequencies: pd.DataFrame
    most_disagreement: pd.DataFrame


@dataclass
class PerturbationResult:
    patient_idx: int
    top_feature: str
    shap_value: float
    original_value: float
    perturbed_to: float
    original_proba: float
    perturbed_proba: float
    proba_change: float


@dataclass
class FaithfulnessResult:
    patients: pd.DataFrame
    direction_match_rate: float


@dataclass
class MedicalDiagnosisResult:
    dataset: MedicalDataset
    split: MedicalSplit
    evaluation: MedicalModel
    shap: ShapResult
    lime: LimeResult
    agreement: AgreementResult
    faithfulness: FaithfulnessResult


class MedicalDiagnosis:
    """Port of medical-diagnosis-xai.ipynb at 86030a3 (moved in 96a568d).

    SHAP and LIME are imported only when explanations are requested. The
    perturbation direction rate preserves the notebook's sign-inequality
    heuristic (including zero changes); it is not a clinical validation.
    """

    DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
    COLUMNS = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    def __init__(
        self, *, random_state: int, test_size: float, stratify: bool,
        data_source: str | Path = DATA_URL, n_estimators: int = 300,
        max_depth: int = 6, n_samples: int = 50, top_k: int = 3,
        lime_num_samples: int = 5000,
    ) -> None:
        if n_samples < 1 or not 1 <= top_k <= len(self.COLUMNS) - 1:
            raise ValueError("n_samples must be positive and top_k must be between 1 and 8")
        if lime_num_samples < 2:
            raise ValueError("lime_num_samples must be at least 2")
        self.random_state = random_state
        self.test_size = test_size
        self.stratify = stratify
        self.data_source = data_source
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.top_k = top_k
        self.lime_num_samples = lime_num_samples

    def load_dataset(self) -> MedicalDataset:
        return self.explore_data(pd.read_csv(self.data_source, names=self.COLUMNS))

    def explore_data(self, frame: pd.DataFrame) -> MedicalDataset:
        frame = frame[self.COLUMNS].copy()
        return MedicalDataset(
            frame, frame.shape, frame.describe(),
            frame.Outcome.value_counts(normalize=True).rename("proportion"),
            (frame[self.ZERO_AS_MISSING] == 0).sum().rename("num_zero_values"),
            frame.corr(numeric_only=True),
        )

    def preprocess(self, dataset: MedicalDataset) -> MedicalSplit:
        clean = dataset.frame.copy()
        clean[self.ZERO_AS_MISSING] = clean[self.ZERO_AS_MISSING].replace(0, np.nan)
        target = clean.Outcome
        X_train, X_test, y_train, y_test = train_test_split(
            clean.drop(columns="Outcome"), target, test_size=self.test_size,
            random_state=self.random_state, stratify=target if self.stratify else None,
        )
        medians = X_train.median()
        if medians.isna().any():
            raise ValueError("Cannot impute a feature with no observed training values")
        return MedicalSplit(X_train.fillna(medians), X_test.fillna(medians), y_train, y_test, medians)

    def train_model(self, split: MedicalSplit) -> MedicalModel:
        model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(split.X_train, split.y_train)
        return self.evaluate_model(model, split)

    def evaluate_model(self, model: RandomForestClassifier, split: MedicalSplit) -> MedicalModel:
        if list(model.classes_) != [0, 1]:
            raise ValueError("Medical diagnosis requires both binary classes 0 and 1")
        predictions = model.predict(split.X_test)
        probabilities = model.predict_proba(split.X_test)[:, 1]
        return MedicalModel(
            model, float(accuracy_score(split.y_test, predictions)),
            float(f1_score(split.y_test, predictions, zero_division=0)),
            float(roc_auc_score(split.y_test, probabilities)),
            pd.DataFrame(classification_report(split.y_test, predictions, output_dict=True, zero_division=0)).T,
            confusion_matrix(split.y_test, predictions, labels=[0, 1]), predictions, probabilities,
        )

    def explain_shap(self, split: MedicalSplit, evaluation: MedicalModel) -> ShapResult:
        import shap

        explanation = shap.TreeExplainer(evaluation.model)(split.X_test)
        return ShapResult(explanation[:, :, 1])

    def explain_lime(
        self, split: MedicalSplit, evaluation: MedicalModel, patient_idx: int = 0,
        explainer: "LimeTabularExplainer | None" = None,
    ) -> LimeResult:
        from lime.lime_tabular import LimeTabularExplainer

        if not 0 <= patient_idx < len(split.X_test):
            raise IndexError("patient_idx must be a position in the test split")
        if explainer is None:
            explainer = LimeTabularExplainer(
                training_data=split.X_train.to_numpy(),
                feature_names=list(split.X_train.columns),
                class_names=["no_diabetes", "diabetes"],
                discretize_continuous=True, random_state=self.random_state,
            )
        explanation = explainer.explain_instance(
            split.X_test.iloc[patient_idx].to_numpy(),
            lambda values: evaluation.model.predict_proba(
                pd.DataFrame(values, columns=split.X_train.columns)
            ),
            labels=(1,), num_features=split.X_train.shape[1], num_samples=self.lime_num_samples,
        )
        return LimeResult(
            explainer, explanation, patient_idx, split.X_test.iloc[patient_idx].copy(),
            int(split.y_test.iloc[patient_idx]), float(evaluation.probabilities[patient_idx]),
            explanation.as_list(label=1),
        )

    def top_k_features_shap(self, values: np.ndarray, names: Sequence[str]) -> FeatureRanking:
        return FeatureRanking([names[i] for i in np.argsort(-np.abs(values))[:self.top_k]])

    def top_k_features_lime(self, explanation: "LimeExplanation", names: Sequence[str]) -> FeatureRanking:
        # Feature IDs avoid ambiguous substring matching in LIME's condition text.
        ranked = sorted(explanation.as_map()[1], key=lambda pair: -abs(pair[1]))
        return FeatureRanking([names[i] for i, _ in ranked[:self.top_k]])

    def compare_explanations(
        self, split: MedicalSplit, evaluation: MedicalModel, shap_result: ShapResult,
        lime_result: LimeResult,
    ) -> AgreementResult:
        indices = np.random.RandomState(self.random_state).choice(
            len(split.X_test), size=min(self.n_samples, len(split.X_test)), replace=False,
        )
        names = list(split.X_train.columns)
        rows = []
        for idx in indices:
            shap_top = set(self.top_k_features_shap(shap_result.positive_class.values[idx], names).features)
            lime = self.explain_lime(split, evaluation, int(idx), lime_result.explainer)
            lime_top = set(self.top_k_features_lime(lime.explanation, names).features)
            rows.append({
                "patient_idx": int(idx), "shap_top_k": sorted(shap_top),
                "lime_top_k": sorted(lime_top), "overlap": sorted(shap_top & lime_top),
                "jaccard": len(shap_top & lime_top) / len(shap_top | lime_top),
            })
        patients = pd.DataFrame(rows)
        frequencies = pd.DataFrame({
            "SHAP": pd.Series([f for row in rows for f in row["shap_top_k"]]).value_counts(normalize=True),
            "LIME": pd.Series([f for row in rows for f in row["lime_top_k"]]).value_counts(normalize=True),
        }).fillna(0).sort_values("SHAP", ascending=False)
        return AgreementResult(
            patients, float(patients.jaccard.mean()), frequencies,
            patients.sort_values("jaccard").head(5),
        )

    def perturb_top_feature_to_median(
        self, patient_idx: int, split: MedicalSplit, evaluation: MedicalModel,
        shap_result: ShapResult,
    ) -> PerturbationResult:
        row = split.X_test.iloc[[patient_idx]].astype(float).copy()
        values = shap_result.positive_class.values[patient_idx]
        feature_idx = int(np.argmax(np.abs(values)))
        feature = str(split.X_test.columns[feature_idx])
        original = float(row[feature].iloc[0])
        median = float(split.X_train[feature].median())
        before = float(evaluation.model.predict_proba(row)[0, 1])
        row[feature] = median
        after = float(evaluation.model.predict_proba(row)[0, 1])
        return PerturbationResult(
            patient_idx, feature, float(values[feature_idx]), original, median, before, after, after - before,
        )

    def run_faithfulness(
        self, split: MedicalSplit, evaluation: MedicalModel, shap_result: ShapResult,
        patient_indices: Sequence[int],
    ) -> FaithfulnessResult:
        if len(patient_indices) == 0:
            raise ValueError("At least one patient is required")
        patients = pd.DataFrame([
            asdict(self.perturb_top_feature_to_median(int(idx), split, evaluation, shap_result))
            for idx in patient_indices
        ])
        matches = np.sign(patients.shap_value) != np.sign(patients.proba_change)
        return FaithfulnessResult(patients, float(matches.mean()))

    def run(self, frame: pd.DataFrame | None = None, patient_idx: int = 0) -> MedicalDiagnosisResult:
        dataset = self.load_dataset() if frame is None else self.explore_data(frame)
        split = self.preprocess(dataset)
        evaluation = self.train_model(split)
        shap_result = self.explain_shap(split, evaluation)
        lime = self.explain_lime(split, evaluation, patient_idx)
        agreement = self.compare_explanations(split, evaluation, shap_result, lime)
        faithfulness = self.run_faithfulness(
            split, evaluation, shap_result, agreement.patients.patient_idx.tolist(),
        )
        return MedicalDiagnosisResult(dataset, split, evaluation, shap_result, lime, agreement, faithfulness)

    def plot_results(self, result: MedicalDiagnosisResult) -> FigureResult:
        """Return the notebook's correlation, SHAP, LIME, and perturbation plots."""
        import matplotlib.pyplot as plt
        import shap

        figures = {}
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        corr = result.dataset.correlations
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)), corr.columns)
        fig.colorbar(image, ax=ax)
        ax.set_title("Feature correlation matrix")
        fig.tight_layout()
        figures["correlations"] = fig
        # SHAP/LIME use pyplot internally; close returned figures to avoid display.
        fig = plt.figure()
        shap.plots.beeswarm(result.shap.positive_class, show=False)
        figures["shap_summary"] = fig
        plt.close(fig)
        fig = plt.figure()
        shap.plots.waterfall(result.shap.positive_class[result.lime.patient_idx], show=False)
        figures["shap_patient"] = fig
        plt.close(fig)
        fig = result.lime.explanation.as_pyplot_figure(label=1)
        fig.tight_layout()
        figures["lime_patient"] = fig
        plt.close(fig)
        fig = Figure(figsize=(6, 5))
        ax = fig.subplots()
        patients = result.faithfulness.patients
        ax.scatter(patients.shap_value, -patients.proba_change, alpha=0.7)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set(xlabel="SHAP value of top feature", ylabel="Negative probability change after perturbation",
               title="Faithfulness: perturbing the top feature to its training median")
        fig.tight_layout()
        figures["faithfulness"] = fig
        return FigureResult(figures)
