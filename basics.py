import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset and train the model
data = pd.read_csv("data.csv") # Use your cleaned dataset here
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # Dropping unnecessary columns
data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)  # Encode diagnosis

# Define target variable and predictors
y = data["diagnosis"]
X = data.drop("diagnosis", axis=1)

# Standardize the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_scaled, y)

# Streamlit Front End
st.title("Breast Cancer Prediction Model")
st.write("This web app predicts the likelihood of a breast tumor being malignant based on user-provided features.")

# Creating input fields for user-defined data
st.header("Input Tumor Features")
st.write("Enter the following measurements:")

# Input fields for each feature
input_data = []
for feature in X.columns:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.5f")
    input_data.append(value)

# Convert input data to DataFrame and scale it
if st.button("Predict Diagnosis"):
    # Standardizing input data
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = lr.predict(input_scaled)
    prediction_prob = lr.predict_proba(input_scaled)[0][1]

    # Display Results
    if prediction == 1:
        st.write("Prediction: The tumor is likely **malignant**.")
        st.write(f"Probability of being malignant: {prediction_prob:.2%}")
    else:
        st.write("Prediction: The tumor is likely **benign**.")
        st.write(f"Probability of being malignant: {prediction_prob:.2%}")
