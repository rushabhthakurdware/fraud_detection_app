# Enterprise Fraud Detection System

An end-to-end real-time financial transaction fraud detection platform powered by hybrid machine learning models, FastAPI, React, and Supabase PostgreSQL.

## 🚀 Key Features

* **Hybrid ML Engine**: Combines **XGBoost** (supervised learning on imbalanced data) with **Isolation Forest** (unsupervised anomaly detection).
* **Explainability Driver**: Computes feature importances to highlight top drivers behind flagged transactions.
* **Audit Trail**: Real-time record logging using **Supabase** (PostgreSQL).
* **Interactive Dashboard**: Modern React UI built with Vite and Tailwind CSS.

## 🛠️ Tech Stack

* **Frontend**: React, Vite, Tailwind CSS
* **Backend**: Python, FastAPI, Pydantic, Scikit-Learn, XGBoost, Joblib
* **Database**: Supabase (PostgreSQL)

## 🏎️ Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload