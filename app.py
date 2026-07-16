import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from models.preprocessing import FEATURE_COLUMNS, build_input_frame, prepare_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "telecom_churn.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
HISTORY_PATH = BASE_DIR / "predictions_history.json"

app = Flask(__name__)
app.secret_key = "connecttel-secret"


def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("The model files are missing. Please run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    config = joblib.load(SCALER_PATH)
    return model, config


def load_dataset():
    return pd.read_csv(DATA_PATH)


def load_history():
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_history(entry: dict) -> None:
    history = load_history()
    history.append(entry)
    with HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def get_risk_band(probability: float) -> tuple[str, str, str]:
    if probability >= 0.8:
        return "High Risk", "danger", "Offer a 20% discount and assign the retention team immediately."
    if probability >= 0.5:
        return "Medium Risk", "warning", "Recommend an upgrade or service review to improve customer satisfaction."
    return "Low Risk", "success", "Customer appears stable; maintain regular engagement and loyalty offers."


def summarize_dashboard(data: pd.DataFrame) -> dict:
    churn_count = int((data["Churn"] == "Yes").sum())
    active_count = int(len(data) - churn_count)
    retention_rate = round((active_count / len(data)) * 100, 1)
    monthly_revenue = round(float(data["MonthlyCharges"].mean() * len(data)), 2)
    avg_tenure = round(float(data["tenure"].mean()), 1)
    return {
        "total_customers": len(data),
        "active_customers": active_count,
        "churn_customers": churn_count,
        "retention_rate": retention_rate,
        "average_monthly_revenue": monthly_revenue,
        "average_tenure": avg_tenure,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    data = load_dataset()
    summary = summarize_dashboard(data)
    return render_template("dashboard.html", summary=summary)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    history = load_history()
    if request.method == "GET":
        return render_template("prediction.html", history=history)

    try:
        form_data = {
            "gender": request.form.get("gender", "Female"),
            "SeniorCitizen": request.form.get("seniorCitizen", "No"),
            "Partner": request.form.get("partner", "No"),
            "Dependents": request.form.get("dependents", "No"),
            "tenure": float(request.form.get("tenure", 12)),
            "PhoneService": request.form.get("phoneService", "Yes"),
            "MultipleLines": request.form.get("multipleLines", "No"),
            "InternetService": request.form.get("internetService", "Fiber optic"),
            "OnlineSecurity": request.form.get("onlineSecurity", "No"),
            "OnlineBackup": request.form.get("onlineBackup", "No"),
            "DeviceProtection": request.form.get("deviceProtection", "No"),
            "TechSupport": request.form.get("techSupport", "No"),
            "StreamingTV": request.form.get("streamingTV", "No"),
            "StreamingMovies": request.form.get("streamingMovies", "No"),
            "Contract": request.form.get("contract", "Month-to-month"),
            "PaperlessBilling": request.form.get("paperlessBilling", "Yes"),
            "PaymentMethod": request.form.get("paymentMethod", "Electronic check"),
            "MonthlyCharges": float(request.form.get("monthlyCharges", 60)),
            "TotalCharges": float(request.form.get("totalCharges", 600)),
        }
        frame = build_input_frame(form_data)
        model, _ = load_model()
        probability = float(model.predict_proba(frame[FEATURE_COLUMNS])[0, 1])
        risk_level, badge_class, action_text = get_risk_band(probability)
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "customerID": request.form.get("customerID", "CUST001"),
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "action": action_text,
        }
        save_history(entry)
        return render_template(
            "prediction.html",
            history=load_history(),
            prediction=entry,
            probability=round(probability * 100, 1),
            risk_level=risk_level,
            badge_class=badge_class,
            action_text=action_text,
        )
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return render_template("prediction.html", history=history)


@app.route("/upload", methods=["POST"])
def upload_batch():
    history = load_history()
    file = request.files.get("batchFile")
    if not file:
        flash("Please upload a CSV file.")
        return render_template("prediction.html", history=history)

    try:
        data = pd.read_csv(file)
        model, _ = load_model()
        prepared = prepare_features(data)
        probabilities = model.predict_proba(prepared[FEATURE_COLUMNS])[0:, 1]
        previews = []
        for idx, probability in enumerate(probabilities):
            risk_level, _, action_text = get_risk_band(float(probability))
            previews.append({"customer": data.iloc[idx].get("customerID", f"CUST{idx + 1}"), "probability": round(float(probability), 4), "risk_level": risk_level, "action": action_text})
        return render_template("prediction.html", history=load_history(), batch_result=previews)
    except Exception as exc:
        flash(f"Batch upload failed: {exc}")
        return render_template("prediction.html", history=history)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("username") == "admin" and request.form.get("password") == "connecttel":
            flash("Welcome back, admin.")
            return redirect(url_for("dashboard"))
        flash("Invalid admin credentials.")
    return render_template("login.html")


@app.route("/search")
def search_customer():
    query = request.args.get("query", "").strip().lower()
    history = load_history()
    filtered = [entry for entry in history if query in entry.get("customerID", "").lower()]
    return render_template("prediction.html", history=filtered)


@app.route("/static/<path:filename>")
def serve_static(filename: str):
    return send_from_directory(BASE_DIR / "static", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
