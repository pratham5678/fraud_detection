# Online Transaction Fraud Detection

## 📋 Project Overview
This project uses a Decision Tree Classifier to detect fraudulent transactions from an online transaction dataset. It performs data exploration, visualization, and builds a machine learning model to classify transactions as fraudulent or not.

The goal is to demonstrate how transaction data can be analyzed and used to predict fraud patterns using Python libraries such as `pandas`, `numpy`, `plotly`, and `scikit-learn`.

---

## ✅ Features
- Exploratory Data Analysis (EDA) with summary statistics and missing values check.
- Visualization using Plotly's interactive pie chart to show distribution of transaction types.
- Correlation analysis between transaction attributes and fraud occurrence.
- Preprocessing of categorical data by mapping transaction types to numeric values.
- Splitting the dataset into training and testing subsets.
- Model training using `DecisionTreeClassifier`.
- Prediction on new transaction data.
- Model evaluation using accuracy.

---

## 📂 Dataset

The dataset used for this project can be downloaded from the following link:

👉 [Download Online Fraud Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) 

> Alternatively, upload the dataset `onlinefraud.csv` into your working directory before running the code.

---

## 📦 Libraries Used
- `pandas` – for data manipulation and analysis
- `numpy` – for numerical computations
- `plotly` – for interactive data visualization
- `scikit-learn` – for machine learning algorithms and model evaluation

---

## ⚙ How to Run
1. Clone or download this repository.
2. Ensure the dataset file `onlinefraud.csv` is placed in the same directory as the script.
3. Install the required libraries if you haven't already:

   ```bash
   pip install pandas numpy plotly scikit-learn
