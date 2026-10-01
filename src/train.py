import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "loan_approval_1000.csv"
)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "loan_approval_model.pkl"
)

df = pd.read_csv(DATA_PATH)
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]


numeric_features = [
    "Age",
    "Income",
    "Loan_Amount",
    "Credit_Score"
]

categorical_features = [
    "Employment_Type"
]


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nModel Evaluation")
print("-----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nClassification Report")
print("---------------------")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Approved", "Approved"]
    )
)

print("Confusion Matrix")
print("----------------")

print(confusion_matrix(y_test, y_pred))


joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully.")
print(MODEL_PATH)