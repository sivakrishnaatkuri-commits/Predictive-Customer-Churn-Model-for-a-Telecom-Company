import os
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.preprocessing import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, prepare_features

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "telecom_churn.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def generate_dataset(path: Path) -> pd.DataFrame:
    np.random.seed(42)
    n = 1200
    df = pd.DataFrame({
        "customerID": [f"CUST{i:03d}" for i in range(1, n + 1)],
        "gender": np.random.choice(["Female", "Male"], size=n, p=[0.52, 0.48]),
        "SeniorCitizen": np.random.choice(["No", "Yes"], size=n, p=[0.82, 0.18]),
        "Partner": np.random.choice(["No", "Yes"], size=n, p=[0.48, 0.52]),
        "Dependents": np.random.choice(["No", "Yes"], size=n, p=[0.63, 0.37]),
        "tenure": np.random.randint(1, 72, size=n),
        "PhoneService": np.random.choice(["No", "Yes"], size=n, p=[0.08, 0.92]),
        "MultipleLines": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.45, 0.35, 0.20]),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.40, 0.45, 0.15]),
        "OnlineSecurity": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.60, 0.25, 0.15]),
        "OnlineBackup": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.55, 0.30, 0.15]),
        "DeviceProtection": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.60, 0.25, 0.15]),
        "TechSupport": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.60, 0.25, 0.15]),
        "StreamingTV": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.55, 0.30, 0.15]),
        "StreamingMovies": np.random.choice(["No", "Yes", "No internet service"], size=n, p=[0.55, 0.30, 0.15]),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]),
        "PaperlessBilling": np.random.choice(["No", "Yes"], size=n, p=[0.45, 0.55]),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n, p=[0.35, 0.20, 0.25, 0.20]),
    })
    df["MonthlyCharges"] = np.clip(np.random.normal(65, 20, n), 18, 120)
    df["TotalCharges"] = np.maximum(0, df["MonthlyCharges"] * df["tenure"] + np.random.normal(0, 120, n))
    churn_prob = (
        (df["Contract"] == "Month-to-month").astype(int) * 0.35
        + (df["PaperlessBilling"] == "Yes").astype(int) * 0.12
        + (df["InternetService"] == "Fiber optic").astype(int) * 0.20
        + (df["TechSupport"] == "No").astype(int) * 0.18
        + (df["SeniorCitizen"] == "Yes").astype(int) * 0.12
        + (df["tenure"] < 12).astype(int) * 0.10
        + (df["MonthlyCharges"] > 70).astype(int) * 0.10
    )
    df["Churn"] = np.random.rand(n) < np.clip(churn_prob, 0.05, 0.85)
    df["Churn"] = df["Churn"].map({True: "Yes", False: "No"})
    df.to_csv(path, index=False)
    return df


def save_plot(filename: str, plot_func) -> None:
    plt.figure(figsize=(8, 5))
    plot_func()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=200)
    plt.close()


def build_visualizations(data: pd.DataFrame) -> None:
    data = data.copy()
    data["Churn"] = data["Churn"].astype(str)
    save_plot("churn_distribution.png", lambda: sns.countplot(x="Churn", data=data, palette="Blues_d"))
    save_plot("gender_distribution.png", lambda: sns.countplot(x="gender", data=data, palette="viridis"))
    save_plot("contract_distribution.png", lambda: sns.countplot(x="Contract", data=data, palette="magma"))
    save_plot("internet_distribution.png", lambda: sns.countplot(x="InternetService", data=data, palette="rocket"))
    save_plot("monthly_charges.png", lambda: sns.histplot(data["MonthlyCharges"], bins=20, kde=True))
    save_plot("total_charges.png", lambda: sns.histplot(data["TotalCharges"], bins=20, kde=True))
    save_plot("tenure_distribution.png", lambda: sns.histplot(data["tenure"], bins=20, kde=True))

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr_data = data.copy()
    corr_data["Churn"] = corr_data["Churn"].replace({"No": 0, "Yes": 1}).astype(int)
    corr = corr_data[numeric_cols + ["Churn"]].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    feature_frame = prepare_features(data)
    feature_frame["Churn"] = feature_frame["Churn"].astype(int)
    model_data = feature_frame[FEATURE_COLUMNS + [TARGET_COLUMN]]
    X = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_COLUMNS),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLUMNS),
        ]
    )
    tree_pipeline = Pipeline([("preprocess", preprocessor), ("classifier", DecisionTreeClassifier(random_state=42))])
    tree_pipeline.fit(X, y)
    importance = pd.Series(
        tree_pipeline.named_steps["classifier"].feature_importances_,
        index=tree_pipeline.named_steps["preprocess"].get_feature_names_out(),
    ).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importance.values[:10], y=importance.index[:10], palette="Blues_d")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=200)
    plt.close()


def train_models(data: pd.DataFrame):
    feature_frame = prepare_features(data)
    model_data = feature_frame[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    X = model_data[FEATURE_COLUMNS]
    y = model_data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_COLUMNS),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLUMNS),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=4000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
    except Exception:
        pass

    results = []
    best_model = None
    best_name = None
    best_score = -1

    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC AUC": round(roc_auc, 4),
        })
        if accuracy > best_score:
            best_score = accuracy
            best_model = pipeline
            best_name = name

    comparison_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    comparison_df.to_csv(IMAGES_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=comparison_df, x="Accuracy", y="Model", palette="Blues_d")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "model_accuracy.png", dpi=200)
    plt.close()

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump({"numeric_columns": NUMERIC_COLUMNS, "categorical_columns": CATEGORICAL_COLUMNS, "feature_columns": FEATURE_COLUMNS}, SCALER_PATH)

    print("Training complete")
    print(comparison_df.to_string(index=False))
    return best_model, best_name, comparison_df


def main() -> None:
    if not DATA_PATH.exists():
        generate_dataset(DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    build_visualizations(data)
    train_models(data)


if __name__ == "__main__":
    main()
