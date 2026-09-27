# XAI Project Journal Entry — Dataset Selection

**Date:** 27 September 2026  
**Project:** Gini vs. Entropy Decision Trees on Medical Datasets

## What I did

Following our group meeting on 21 September, I was assigned the task of finding suitable datasets for our combined XAI research topic and explaining why those datasets would be appropriate.

Our original ideas were to investigate either:

- the use of SHAP and LIME for explaining machine-learning models in a medical context; or
- the difference between using Gini impurity and entropy/information gain when constructing decision trees.

We ultimately agreed to combine the ideas. The project will use medical classification datasets and build decision-tree models using both Gini impurity and entropy as the splitting criteria. We can then compare the resulting models in terms of both their predictive performance and interpretability.

I therefore focused my dataset search on medical datasets that would work well for classification and, importantly, would allow us to meaningfully investigate the structure and interpretability of decision trees.

## Dataset selection criteria

I decided that a suitable dataset should have the following characteristics:

1. **A classification problem** — our models need a clear target variable that can be predicted using a decision tree.
2. **A medical or healthcare context** — this keeps the experiment aligned with the project's proposed application area.
3. **Tabular data** — decision trees are particularly suitable for this type of structured data.
4. **Meaningful features** — variables should have understandable medical interpretations so that we can discuss whether the resulting decision trees are interpretable.
5. **A reasonable number of observations** — the datasets should contain enough examples to make comparisons useful without being unnecessarily large for the scope of the project.
6. **A manageable number of features** — this makes the resulting trees easier to analyse and visualise.
7. **Reliable provenance** — I preferred established datasets such as those provided through the UCI Machine Learning Repository, rather than choosing datasets solely because they were easily available on Kaggle.
8. **Different characteristics between datasets** — using more than one dataset should allow us to investigate whether any observed differences between Gini and entropy are consistent across different medical problems.

I identified three datasets that I think will best support our research.

### 1. Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

- 569 instances
- 30 numerical features
- Binary classification
- Features are measurements derived from digitised images of breast masses
- Target relates to whether the tumour is malignant or benign

I selected this dataset because it provides a relatively large number of observations while remaining small enough to work with easily. It also contains 30 numerical features, giving the decision tree many possible variables from which to construct splits.

The medical meaning of the classification problem is clear, and the dataset is an established machine-learning benchmark with substantial existing research. This gives us useful background literature and makes it easier to justify the dataset in the final report.

It should also be useful when comparing Gini and entropy because we can examine whether the two criteria select different features, produce different tree structures, or result in different levels of predictive performance.

### 2. Heart Disease

**Source:** UCI Machine Learning Repository

- 303 instances in the commonly used processed version
- 13 features
- Classification target relating to the presence of heart disease
- Contains numerical and categorical variables

Examples of features include age, sex, chest-pain type, resting blood pressure, cholesterol, maximum heart rate and exercise-induced angina.

I chose this dataset because the features are particularly easy to understand from an interpretability perspective. If a decision tree uses variables such as age, blood pressure or chest-pain type, we can explain what those decisions represent rather than simply discussing anonymous numerical features.

It also provides a useful contrast with the Breast Cancer dataset because it contains a smaller number of features and includes different types of variables. This gives us another environment in which to compare the two splitting criteria.

One issue to document will be the presence of missing values and the preprocessing required before training the models.

### 3. Heart Failure Clinical Records

**Source:** UCI Machine Learning Repository

- 299 instances
- 12 features
- Medical/clinical data from heart-failure patients
- No missing values according to the UCI dataset description

The dataset contains variables such as age, anaemia, diabetes, high blood pressure, ejection fraction, serum creatinine, serum sodium, smoking and sex.

I selected this dataset because it is relatively small and clean while still containing medically meaningful variables. The variables should make the resulting decision trees relatively straightforward to interpret.

The absence of missing values also makes it useful as a relatively clean baseline dataset. This means that differences observed between Gini and entropy are less likely to be complicated by missing-data handling.

## How this will lead into the experiment

The datasets will be used to train decision-tree classifiers under two different splitting criteria:

- **Gini impurity**
- **Entropy / information gain**

## Next steps

The next stage should be to prepare the selected datasets for experimentation.

Planned steps are:

1. Obtain the datasets from their original/reliable sources.
2. Inspect the features and target variables.
3. Identify and document missing values and other preprocessing requirements.
4. Determine how categorical variables should be encoded where necessary.
5. Establish a consistent train/test methodology.
6. Train equivalent decision trees using Gini and entropy.
7. Record predictive performance metrics.
8. Record tree-structure and interpretability metrics.
9. Visualise the resulting trees where useful.
10. Compare the results across all three datasets.
11. Review relevant research papers on Gini impurity, entropy and decision-tree interpretability.
12. Use the results to inform the final report and presentation.
