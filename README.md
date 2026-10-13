# Customer Churn Probability App

Streamlit application that scores bank customers with a Random Forest model and ranks them by churn risk.

- Live app: <https://churnprobaapps-ftuk5k4wd7umfmm5angxsk.streamlit.app/>
- Project page, notebook and documentation: <https://ldongmon.github.io/> ([repository](https://github.com/ldongmon/ldongmon.github.io))

## What the app does

1. Loads `churn_model.pkl`, a scikit-learn pipeline (scaling, one-hot encoding, Random Forest of 300 trees).
2. Reads an uploaded CSV file of customers.
3. Returns a churn probability and a risk band for each customer, with a downloadable CSV.

## Required columns

Same names as the Kaggle *Bank Customer Churn* dataset. Any other column is kept in the output but ignored by the model.

| Column | Type |
|---|---|
| `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`, `Point Earned` | numeric |
| `HasCrCard`, `IsActiveMember` | 0 or 1 |
| `Satisfaction Score` | 1 to 5 |
| `Geography` | France, Spain, Germany |
| `Gender` | Male, Female |
| `Card Type` | DIAMOND, GOLD, SILVER, PLATINUM |

## Model performance

Measured on a held-out test set of 2,000 customers (408 churners):

| Metric | Value |
|---|---|
| Accuracy | 0.869 (trivial baseline 0.796) |
| Precision (churn) | 0.80 |
| Recall (churn) | 0.47 |
| F1 (churn) | 0.59 |
| ROC-AUC | 0.867 |

The probabilities are not calibrated: they rank customers reliably, but they are not a literal percentage chance of leaving.

## Risk bands

| Band | Probability |
|---|---|
| Low risk | below 0.20 |
| Moderate | 0.20 to 0.40 |
| High risk | 0.40 to 0.70 |
| Critical | 0.70 and above |

## Run locally

```bash
pip install -r requirements.txt
streamlit run churn_proba_apps.py
```

`scikit-learn` is pinned to 1.6.1 because `churn_model.pkl` was saved with that version.

## Contact

Leprince Dongmo Nanda · <leprince.dongmo@yahoo.fr>
