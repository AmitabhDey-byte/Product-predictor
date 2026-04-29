# Product Prediction

<img width="1915" height="868" alt="Screenshot 2026-04-29 180404" src="https://github.com/user-attachments/assets/cf4785b6-64e1-455b-b675-6369d6800696" />
<img width="1917" height="965" alt="Screenshot 2026-04-29 180314" src="https://github.com/user-attachments/assets/73ce6c7b-026d-45ef-a136-db6e302825d7" />


## Overview

This project is a machine learning-based application that predicts product quality and provides smart recommendations based on user input. It combines data preprocessing, model building, feature engineering, and an interactive dashboard.

The system allows users to input product details such as price and rating count, and then:

* Predicts whether the product is likely to be good or bad
* Recommends similar high-quality products using a hybrid recommendation system

---

## Features

* Data preprocessing pipeline
* Baseline model using Logistic Regression
* Improved model using Random Forest
* Feature engineering for better performance
* Model comparison and evaluation
* Hybrid recommendation system (similarity + model filtering)
* Interactive dashboard using Streamlit

---

## Dataset

The dataset used is an Amazon product dataset containing:

* Product name
* Discounted price
* Actual price
* Rating
* Rating count

---

## Feature Engineering

The following features were created to improve model performance:

* Price difference (`actual_price - discounted_price`)
* Discount ratio (`discounted_price / actual_price`)
* Log of rating count (`log1p(rating_count)`)

---

## Models Used

### Model 1: Logistic Regression

* Used as a baseline model
* Provides a simple and interpretable benchmark

### Model 2: Random Forest

* Final selected model
* Performs better due to ensemble learning and ability to capture non-linear patterns

---

## Model Evaluation

* Train / Validation / Test split was used
* Accuracy metric used for evaluation
* Cross-validation and GridSearchCV used for tuning

---

## Recommendation System

A hybrid recommendation system was implemented:

* Uses euclidean distance to find similar products
* Filters results using Random Forest predictions
* Ensures recommended products are both similar and high quality

---

## Dashboard

The project includes a Streamlit dashboard with:

* User input for product details
* Model selection (Logistic Regression or Random Forest)
* Real-time prediction

---

* Improve UI/UX of the dashboard
* Deploy the application online
