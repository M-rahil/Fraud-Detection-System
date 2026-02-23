# Credit Card Fraud Detection System

An end-to-end machine learning application that detects fraudulent credit card transactions using a trained Random Forest classifier and a Streamlit-based web interface.

## Overview

This project demonstrates the complete ML lifecycle:

- Data exploration and preprocessing
- Model training and evaluation
- Model persistence using joblib
- Web deployment using Streamlit
- Batch fraud prediction via CSV upload

## Model Details

- Algorithm: Random Forest Classifier
- Dataset: Credit Card Fraud Detection Dataset (PCA-based features)
- Features: Time, V1–V28, Amount
- Target: Class (0 = Normal, 1 = Fraud)

## Project Structure

Fraud-Detection-System/
│
├── app/app.py
├── model/fraud_model.pkl
├── notebooks/eda.ipynb
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Clone the repository:

git clone https://github.com/your-username/Fraud-Detection-System.git

Install dependencies:

pip install -r requirements.txt

Run the application:

cd app
streamlit run app.py

## Features

- Upload transaction dataset (CSV)
- Fraud prediction for each transaction
- Fraud risk probability scoring
- Downloadable prediction results

## Dataset

The dataset used for training can be found on Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Note: The dataset is not included in this repository due to size limitations.

## Future Improvements

- REST API integration using FastAPI
- Real-time transaction scoring
- Cloud deployment
- Model monitoring and logging
