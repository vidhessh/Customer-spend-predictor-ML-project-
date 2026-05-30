# Customer Spend Prediction & Marketing Intelligence

> Predict how much a customer will spend using an ensemble of 3 ML models,
> then receive targeted marketing action recommendations.

## Tech Stack
Python · Streamlit · Random Forest · XGBoost · Ridge Regression · KMeans · Scikit-learn

## Features
- Predicts customer spend from profile inputs
- Compares 3 models with accuracy scores
- Generates marketing strategy based on predicted spend tier
- Customer segmentation using KMeans clustering

## How to Run
```bash
pip install streamlit scikit-learn xgboost numpy
streamlit run app.py
```

## Model Accuracy
| Model | Accuracy |
|---|---|
| Random Forest | 94.7% |
| XGBoost | 91.3% |
| Ridge Regression | 82.6% |

## Dataset
350 customer records with features: Age, Gender, Membership Type,
Items Purchased, Average Rating, Discount Applied, Days Since Last Purchase.

---
By **Vidhessh**
