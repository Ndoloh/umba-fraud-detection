# Umba Real-Time Fraud Detection System

## Project Overview

This project implements a complete real-time fraud detection system for financial transactions.

The solution includes:

- Data preprocessing and feature engineering
- Machine learning model training
- Fraud prediction API using FastAPI
- Interactive dashboard using Streamlit
- Docker configuration for deployment

---

# Project Structure

```
umba-fraud-detection-main/

├── API/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── final_model.pkl
│   ├── predictions.csv
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── test_transaction.csv
│
├── notebooks/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Project Workflow

## Step 1: Data Preparation

The transaction dataset and identity dataset were loaded into Python.

The identity information was aggregated and merged with the transaction dataset.

Missing values were handled and unnecessary leakage features were removed.

---

## Step 2: Feature Engineering

Several preprocessing steps were applied.

These included:

- Missing value handling
- Encoding categorical variables
- Scaling numerical variables where necessary

---

## Step 3: Model Training

Two machine learning models were trained.

The first model was a Random Forest baseline.

The second model was a LightGBM classifier.

The LightGBM model achieved better fraud detection performance and was selected as the final model.

---

## Step 4: Threshold Selection

The Precision-Recall curve was used to determine the optimal probability threshold.

The best threshold was selected to maximize fraud detection performance.

---

## Step 5: Final Model

The final LightGBM model was retrained using the complete training dataset.

The trained model was saved as:

```
data/final_model.pkl
```

Predictions were generated and saved as:

```
data/predictions.csv
```

---

## Step 6: REST API

A FastAPI application was created.

The API loads the trained model and predicts whether a transaction is fraudulent.

Swagger documentation is automatically available.

---

## Step 7: Dashboard

A Streamlit dashboard was developed.

The dashboard allows users to:

- Enter transaction information
- View fraud predictions
- Display prediction probabilities

---

## Step 8: Docker Deployment

Docker support was added to simplify deployment.

Files included:

- Dockerfile
- docker-compose.yml

---

# Installation

Clone the repository.

```
git clone <repository_url>
```

Move into the project.

```
cd umba-fraud-detection-main
```

Install dependencies.

```
pip install -r requirements.txt
```

---

# Running the API

```
uvicorn API.app:app --reload
```

API:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

# Running the Dashboard

```
streamlit run dashboard/app.py
```

Dashboard:

```
http://localhost:8501
```

---

# Docker

Build the Docker image.

```
docker build -t umba-fraud-detection .
```

Run the API.

```
docker run -p 8000:8000 umba-fraud-detection
```

Run both API and Dashboard.

```
docker compose up --build
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- FastAPI
- Streamlit
- Docker

---

# Outputs

The project generates:

- Trained machine learning model
- Fraud predictions
- REST API
- Interactive dashboard
- Docker deployment files

---

# Author

Ndolo Moses Onyinge

Email: ndolomoses254@gmail.com

GitHub: https://github.com/Ndoloh/umba-fraud-detection