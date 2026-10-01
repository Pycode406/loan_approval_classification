# Loan Approval Classification

A machine learning project that predicts whether a loan application
will be approved using Logistic Regression.

## Dataset

The dataset contains 1000 loan application records.

### Features

- Age
- Income
- Loan Amount
- Credit Score
- Employment Type

### Target

- Loan Approved

Where:

- 0 = Loan Not Approved
- 1 = Loan Approved

## Machine Learning Workflow

1. Data Understanding
2. Data Preprocessing
3. Train/Test Split
4. Logistic Regression
5. Model Training
6. Model Evaluation
7. Model Saving
8. New Record Prediction

## Preprocessing

### Numerical Features

Missing values are handled using median imputation and
numerical features are standardized using StandardScaler.

### Categorical Features

Missing categorical values are replaced using the most frequent
value and Employment Type is converted using OneHotEncoder.

## Model

Logistic Regression is used for binary classification.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Model Saving

The complete preprocessing and Logistic Regression pipeline is
saved using Joblib.

The saved model is:

models/loan_approval_model.pkl

## Running the Project

### Train the model

```bash
python src/train.py
