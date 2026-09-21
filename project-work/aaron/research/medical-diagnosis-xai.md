# Explainable AI for Medical Diagnosis — Initial Research

**Status:** Initial investigation  
**Date:** 21 September 2026  
**Authors:** Aaron McGuinness

---

## 1. Research Topic

### Explainable AI in Medical Diagnosis

One potential direction for our xAI project is investigating how Explainable AI (XAI) can be applied to machine-learning models used for medical diagnosis.

The basic idea would be to train a machine-learning model to predict whether a patient has a particular medical condition, such as heart disease or diabetes. XAI techniques could then be used to explain why the model produced a particular prediction.

For example, rather than simply producing:

> **Prediction:** Heart disease  
> **Probability:** 87%

an XAI method could identify the features that contributed to the prediction, such as age, cholesterol, blood pressure, or chest pain.

We belive this is an interesting topic to consider because understanding the reasons behind a prediction is very important when models are used to support clinical decision-making.

---

## 2. Existing Research

There is already substantial research applying XAI to healthcare and medical diagnosis.

A 2025 systematic literature review examined 30 studies on XAI for disease prediction and identified SHAP and LIME as two commonly used approaches. The review also identified limitations including limited dataset diversity and the need for greater interpretability.  
[The role of explainable artificial intelligence in disease prediction](https://pubmed.ncbi.nlm.nih.gov/40038704/)

A 2026 systematic review examined 36 empirical healthcare XAI studies, including 16 focused specifically on diagnosis. SHAP was used in 21 studies and LIME in approximately 11. The authors noted that studies frequently combine multiple explanation methods and highlighted the need for stronger evaluation of explanation faithfulness and stability.  
[Explainable AI in healthcare: a systematic review of XAI use cases](https://pubmed.ncbi.nlm.nih.gov/41994555/)

Another systematic review of XAI in electronic health-record research found that SHAP was the most frequently used XAI method in the studies examined, while also identifying a lack of critical evaluation of the validity and robustness of explanations.  
[Application of XAI in electronic health record research](https://pubmed.ncbi.nlm.nih.gov/39493635/)

There is also existing work directly applying SHAP and LIME to diabetes prediction, demonstrating that this type of experiment is technically feasible.  
[Explainable AI and Interpretable Machine Learning: A Case Study in Perspective](https://www.sciencedirect.com/science/article/pii/S1877050922008432)

---

## 3. Potential Research Direction

Simply training a medical prediction model and producing SHAP/LIME visualisations would likely reproduce existing work. A more interesting direction could therefore be investigating how reliable the explanations actually are.

For example, we could compare SHAP and LIME explanations for the same predictions and investigate whether they identify the same important features.

We could also investigate faithfulness:

> Does the explanation accurately represent the features that actually influence the model's prediction?

One possible experiment would be to identify features considered important by SHAP or LIME, change those features, and observe whether the model's prediction changes accordingly.

A 2026 paper specifically proposed evaluating healthcare XAI using metrics including fidelity, simplicity, consistency, robustness, precision and coverage, providing a potential basis for such an experiment.  
[Explainability in action: A metric-driven assessment of local explanations for healthcare tabular models](https://pubmed.ncbi.nlm.nih.gov/42418422/)

---

## 4. Initial Assessment

Medical diagnosis appears to be a viable area for further investigation because there is substantial existing research, publicly available datasets, and established XAI techniques such as SHAP and LIME.

The main question for us is whether we can identify a sufficiently specific research question that goes beyond simply applying existing XAI techniques.

Potential directions include:

- Comparing SHAP and LIME
- Investigating explanation faithfulness
- Investigating explanation stability
- Evaluating different metrics for explanation quality
- Investigating whether XAI can identify unexpected model behaviour

Further research and experimentation are required before deciding whether this will become our final project topic.
