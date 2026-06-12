# Credit Card Fraud Detection

## Project Overview
This project detects fraudulent credit card transactions using Machine Learning. A Random Forest Classifier is trained on the Credit Card Fraud Detection dataset to classify transactions as Fraud or Genuine.

## Features
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Class distribution visualization
- Correlation heatmap
- Random Forest model training
- Fraud prediction
- Streamlit web application

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Dataset
The dataset contains credit card transactions with 31 features and a target column named `Class`.

- Class 0 → Genuine Transaction
- Class 1 → Fraudulent Transaction

## Model Performance
- Accuracy: 99.81%
- Precision: 0.47
- Recall: 0.88
- F1-Score: 0.61

## Project Structure

credit-card-fraud-detection/
│
├── fraud_detection.ipynb
├── app.py
├── README.md
├── creditcard.csv
│
└── screenshots/
    ├── dataset_info.png
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── model_accuracy.png
    └── streamlit_app.png

## How to Run

1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

2. Run the application

```bash
streamlit run app.py
```

## Author
Mini Project - Credit Card Fraud Detection