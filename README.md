# ConnectTel Customer Churn Prediction System

A full-stack machine learning web application that predicts customer churn for ConnectTel using Flask and scikit-learn.

## Features
- Responsive landing page, prediction form, dashboard, and about page
- Machine learning pipeline with preprocessing, feature engineering, and model comparison
- Real-time churn prediction and business recommendations
- Batch CSV prediction and prediction history

## Setup
1. Create and activate a virtual environment
   - Windows: `python -m venv venv`
   - `venv\Scripts\activate`
2. Install dependencies
   - `pip install -r requirements.txt`
3. Train the model
   - `python train_model.py`
4. Run the app
   - `python app.py`
5. Open `http://127.0.0.1:5000`

## Project Files
- app.py: Flask application
- train_model.py: Training pipeline and chart generation
- models/preprocessing.py: Data cleaning and feature engineering logic
- templates/: HTML templates
- static/: CSS, JS, and generated images
