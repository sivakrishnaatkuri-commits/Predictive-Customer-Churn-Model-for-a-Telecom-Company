import pandas as pd
import numpy as np


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AverageMonthlySpending"] = df["TotalCharges"] / np.where(df["tenure"] > 0, df["tenure"], 1)
    df["LongTermCustomer"] = (df["tenure"] >= 24).astype(int)
    df["HighUsageCustomer"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)
    df["PaperlessBillingFlag"] = (df["PaperlessBilling"].eq("Yes")).astype(int)
    df["SeniorCitizenCategory"] = (df["SeniorCitizen"].eq("Yes")).astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    )
    df["MonthlyChargesCategory"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 70, 100],
        labels=["Low", "Medium", "High"],
    )
    return df
