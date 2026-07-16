import os
from typing import Dict, List

import numpy as np
import pandas as pd


TARGET_COLUMN = "Churn"
NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizenCategory",
    "PaperlessBillingFlag",
    "LongTermCustomer",
    "HighUsageCustomer",
    "AverageMonthlySpending",
]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "MonthlyChargesCategory",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)

    for column in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["MonthlyCharges"] * cleaned["tenure"])
    cleaned["MonthlyCharges"] = cleaned["MonthlyCharges"].fillna(cleaned["MonthlyCharges"].median())
    cleaned["tenure"] = cleaned["tenure"].fillna(cleaned["tenure"].median())

    for column in ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
            cleaned[column] = cleaned[column].fillna("Unknown")

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].replace({"0": "No", "1": "Yes", "False": "No", "True": "Yes"})
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)
    cleaned["Partner"] = cleaned["Partner"].replace({"No": "No", "Yes": "Yes", "Unknown": "Unknown"})
    cleaned["Dependents"] = cleaned["Dependents"].replace({"No": "No", "Yes": "Yes", "Unknown": "Unknown"})
    cleaned["PhoneService"] = cleaned["PhoneService"].replace({"No": "No", "Yes": "Yes", "Unknown": "Unknown"})
    cleaned["PaperlessBilling"] = cleaned["PaperlessBilling"].replace({"No": "No", "Yes": "Yes", "Unknown": "Unknown"})
    cleaned["Churn"] = cleaned["Churn"].replace({"No": 0, "Yes": 1, "False": 0, "True": 1, "0": 0, "1": 1})
    cleaned["Churn"] = cleaned["Churn"].astype(int)
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["AverageMonthlySpending"] = engineered["TotalCharges"] / np.where(engineered["tenure"] > 0, engineered["tenure"], 1)
    engineered["LongTermCustomer"] = (engineered["tenure"] >= 24).astype(int)
    engineered["HighUsageCustomer"] = (engineered["MonthlyCharges"] >= engineered["MonthlyCharges"].median()).astype(int)
    engineered["PaperlessBillingFlag"] = (engineered["PaperlessBilling"].eq("Yes")).astype(int)
    engineered["SeniorCitizenCategory"] = (engineered["SeniorCitizen"].eq("Yes")).astype(int)
    engineered["TenureGroup"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )
    engineered["MonthlyChargesCategory"] = pd.cut(
        engineered["MonthlyCharges"],
        bins=[0, 35, 70, 100],
        labels=["Low", "Medium", "High"],
    )
    return engineered


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = engineer_features(clean_dataset(df))
    prepared["gender"] = prepared["gender"].astype(str).fillna("Unknown")
    for column in ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup", "MonthlyChargesCategory"]:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype(str).fillna("Unknown")
    return prepared


def build_input_frame(raw_values: Dict[str, object]) -> pd.DataFrame:
    base = {
        "customerID": raw_values.get("customerID", "CUST001"),
        "gender": raw_values.get("gender", "Female"),
        "SeniorCitizen": raw_values.get("SeniorCitizen", "No"),
        "Partner": raw_values.get("Partner", "No"),
        "Dependents": raw_values.get("Dependents", "No"),
        "tenure": float(raw_values.get("tenure", 12)),
        "PhoneService": raw_values.get("PhoneService", "Yes"),
        "MultipleLines": raw_values.get("MultipleLines", "No"),
        "InternetService": raw_values.get("InternetService", "Fiber optic"),
        "OnlineSecurity": raw_values.get("OnlineSecurity", "No"),
        "OnlineBackup": raw_values.get("OnlineBackup", "No"),
        "DeviceProtection": raw_values.get("DeviceProtection", "No"),
        "TechSupport": raw_values.get("TechSupport", "No"),
        "StreamingTV": raw_values.get("StreamingTV", "No"),
        "StreamingMovies": raw_values.get("StreamingMovies", "No"),
        "Contract": raw_values.get("Contract", "Month-to-month"),
        "PaperlessBilling": raw_values.get("PaperlessBilling", "Yes"),
        "PaymentMethod": raw_values.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges": float(raw_values.get("MonthlyCharges", 60)),
        "TotalCharges": float(raw_values.get("TotalCharges", 600)),
        "Churn": 0,
    }
    frame = pd.DataFrame([base])
    return engineer_features(clean_dataset(frame))
