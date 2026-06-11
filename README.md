## Explainable Credit Risk Prediction — Deployed ML Web App

## Overview

This project applies machine learning to predict credit risk using the German Credit dataset. The aim is to classify applicants as good or bad credit risks and to explain the model's results using explainable AI methods.

The project includes data understanding, preprocessing, model training, comparison, evaluation, and explainability analysis.

## Dataset

The dataset used in this project is the German Credit dataset from the UCI Machine Learning Repository.

Dataset link: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

The dataset contains applicant and financial attributes, including credit amount, duration, checking account status, savings account status, employment information, and credit history.

## Project Objectives

* Understand and explore the German Credit dataset.
* Clean and preprocess the data for machine learning.
* Train multiple classification models.
* Compare model results using evaluation metrics.
* Explain model predictions using explainable AI techniques.
* Present the results in a technical report.

## Models Used

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* HistGradientBoosting

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

## Explainability

Explainability was included to make model results easier to interpret and more useful in a real-world credit risk setting.

The project includes:

* Logistic Regression coefficient analysis
* Random Forest feature importance
* SHAP summary plot
* SHAP bar plot
* SHAP waterfall plots for individual applicants

## Project Structure

## Project Structure

```text
explainable-credit-risk-prediction/
│
├── data/
│   ├── raw/
│   │   ├── german.data
│   │   └── german.doc
│   │
│   └── processed/
│       └── german_credit_clean.csv
│
├── figures/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_and_baseline.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_explainability.ipynb
│
├── reports/
│   └── Abedalaziz_Credit_Risk_Report.docx
│
├── .gitignore
├── README.md
└── requirements.txt
```
