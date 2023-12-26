# Credit Risk Analysis & Prediction


# Overview

This project involves analyzing credit risk using the German Credit Data dataset. The objective is to predict the likelihood of customers being good or bad credit risks based on their demographic and account information.

# Dataset

The dataset german_credit_data.csv.xls contains various attributes such as age, job, housing, saving accounts, checking account, credit amount, duration, and purpose along with a target variable 'Risk' indicating good or bad credit.

# Features

Exploratory Data Analysis (EDA): Initial exploration of data using Python's seaborn and matplotlib libraries for visualizing distributions and relationships.
Data Preprocessing: Conversion of data types and handling missing values.
Feature Engineering: Derivation of new features to improve the model's predictive power.
Statistical Analysis: Use of violin plots, box plots, and histograms to understand the data distribution and relationships.
Predictive Modeling: Application of various classification algorithms like Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forest, XGBoost, and LightGBM.
Model Evaluation: Using cross-validation, accuracy score, confusion matrix, and F-beta score for evaluating model performance.
Hyperparameter Tuning: Utilization of GridSearchCV for optimizing model parameters.
Pipeline Creation: Integration of feature selection and model training in a single pipeline.
Requirements

# This project requires the following Python libraries:

pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
xgboost
lightgbm
catboost
