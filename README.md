# 🏡 California Housing Price Prediction — End-to-End ML Project

This repository contains a complete Machine Learning pipeline built using the California Housing dataset, with the goal of predicting median house values using demographic and housing-related features.

The project includes data exploration, preprocessing, model training, prediction generation, and visual evaluation.

# 📁 Repository Contents

# 📊 1. housing.csv

Full dataset used for:

data exploration

preprocessing

stratified sampling

training and evaluation

# 💻 2. main.py

Python script that:

loads the processed dataset

trains the machine learning model

saves the trained model as a .joblib file

(Optional) generates predictions

Useful if you want to run the project without the notebook.

# 📘 3. process_with_explaination.ipynb

The full Jupyter Notebook showing:

entire ML workflow

preprocessing steps

pipelines

handling missing values

model comparisons

visualization (scatter plots, metrics, etc.)

explanations for each step

This is the core learning component of the project.

# 📄 4. predictions.csv

Contains model predictions for the test data.

Useful for:

visualizing predicted vs actual values

storing model inference output

referencing model performance externally

# 🧪 5. testing.csv

20% test split of the total dataset.
Used for final evaluation after training.

# 🚀 Project Overview

This project demonstrates an end-to-end supervised regression pipeline using Scikit-Learn. It walks through:

✔ Data Exploration

correlations

histograms

geographic visualizations

outlier analysis

✔ Data Preprocessing

handling missing values

encoding categorical attributes

numerical transformations

building reusable pipelines

stratified sampling based on income categories

✔ Model Training

Several regression models were trained and compared:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

✔ Model Evaluation

RMSE calculation

cross-validation

actual vs predicted scatter plots

feature importance analysis

✔ Saving the Model

The trained model was saved using joblib inside main.py.

Note: The actual .joblib model file is NOT included because it exceeds GitHub’s size limit and is unnecessary for understanding or reproducing the project.

📸 Visualizations

The notebook includes helpful visualizations such as:

correlation matrix

scatter plot of Actual vs Predicted values

histograms

geographic price map

These help illustrate model performance and dataset characteristics.

# 🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-Learn

Joblib

Jupyter Notebook

# ▶️ How to Run the Project
Option 1 — Run the Notebook
jupyter notebook process_with_explaination.ipynb

Option 2 — Run main.py
python main.py


This will train the model and generate the .joblib file.

# 📌 Notes

The trained model file is intentionally excluded to keep the repository lightweight.

Anyone can retrain the model from the notebook or main.py.

predictions.csv and testing.csv are provided for convenience.

# ⭐ If you found this project helpful, consider leaving a star!
