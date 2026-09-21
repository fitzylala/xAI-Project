## Project Proposal - Skye 21/09/2026

**"Does the choice of splitting criterion (Gini impurity vs entropy/information gain) affect the interpretability of decision trees?"**

---

### Background & Motivation

In class, we've worked through the maths of both splitting criteria:

- **Entropy & Information Gain**: rooted in information theory (Shannon), measures the reduction in uncertainty
- **Gini Impurity**: simpler computationally, measures the probability of misclassification

While we understand the maths and working out behind each approach, the outcomes for model interpretability aren't clear. I was talking to James about what the motivation would be for using either one of these above the other, and he says that Gini is usually set by default but he's not sure why - and that it would be an interesting topic to research for the xAI project.

**Why this matters:** Decision trees are already considered interpretable, but interpretability isn't binary. If splitting criteria produce different tree structures (and different interpretability characteristics), this could inform best practices for deploying decision trees in whatever domain.

---

### Research Work Scope & Methodology

#### Phase 1: Theoretical Foundation

- Document the mathematical origins and properties of both criteria
- Compare computational complexity
- Survey existing papers on whether researchers have observed differences in tree structure
- Clarify the relationship between these metrics and interpretability theory

#### Phase 2: Empirical Evaluation

Select 5-8 public datasets spanning:

- **Size diversity**: small (< 1000 rows), medium (1K-10K), large (> 10K)
- **Domain diversity**: tabular classification tasks (UCI ML Repository, Kaggle)
- **Feature types**: mixed numeric/categorical, all numeric, etc.

For each dataset, train decision trees using both criteria with consistent hyperparameters.

#### Phase 3: Interpretability Assessment

Measure interpretability across multiple dimensions:

**Structural Complexity:**

- Tree depth (max and mean)
- Number of nodes and leaves
- Total splits required to classify instances
- _Rationale_: Shallower, simpler trees are easier for humans to trace through

**Predictive Stability:**

- Train on multiple random 80/20 splits of the same dataset
- Track whether the top 3-5 splits remain consistent across runs
- Calculate Kendall's Tau correlation of feature importance rankings
- _Rationale_: Unstable trees suggest the model is fragile; humans can't trust explanations if the model itself is unreliable

**Accuracy-Interpretability Tradeoff:**

- Compare accuracy when both trees are constrained to the same depth (3, 5, 7, 10)
- Which criterion achieves higher accuracy at equal "complexity budgets"?
- _Rationale_: If one criterion is more accurate at shallow depths, it's more interpretable per unit of performance

**Feature Coverage & Sparsity:**

- Count unique features used in each tree
- Track feature usage patterns (which features dominate?)
- _Rationale_: Models using fewer features are conceptually simpler

**Visual Interpretability:**

- Generate tree visualizations for comparison
- Assess subjective clarity (feature names, split thresholds, decision rules at leaves)

### Expected Outcomes

**Null Hypothesis:** No significant systematic difference in interpretability between criteria

**Alternative Hypotheses:**

1. Gini produces structurally simpler trees (fewer splits) but less stable feature importance
2. Entropy produces more stable, consistent feature rankings but deeper trees
3. The difference is dataset-dependent (criterion choice interacts with data characteristics)

### Deliverables

1. **Jupyter notebook** with end-to-end analysis
2. **Summary report** comparing metrics across datasets
3. **Visualizations**: tree comparisons, metric plots, stability analysis
4. **Recommendations**: when to prefer one criterion over the other for interpretability-focused applications
5. **Code repository** with reproducible experiments

### Success Criteria

- Clear empirical evidence on how splitting criteria affect at least 3+ of the interpretability metrics
- Actionable insights for practitioners choosing splitting criteria
- Well-documented, reproducible experiments across multiple datasets
- Honest assessment of limitations (e.g., "interpretability is subjective")
