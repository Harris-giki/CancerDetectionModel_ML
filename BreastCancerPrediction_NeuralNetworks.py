import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('breast_cancer_model.h5')  # Replace with your model file path

# Initialize the scaler (use the same scaling method as used during training)
scaler = StandardScaler()

# Input feature names
feature_names = [
    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
    'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry',
    'Mean Fractal Dimension', 'Radius Error', 'Texture Error', 'Perimeter Error',
    'Area Error', 'Smoothness Error', 'Compactness Error', 'Concavity Error',
    'Concave Points Error', 'Symmetry Error', 'Fractal Dimension Error',
    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
    'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry',
    'Worst Fractal Dimension'
]

# Streamlit App
st.title("Breast Cancer Prediction Application")
st.write("""
This app predicts whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) 
based on clinical features. Please input the required values below.
""")

# Create input fields for user to enter data
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, max_value=1000.0, step=0.01)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    # Convert user input into a NumPy array and reshape it
    input_data = np.array(user_input).reshape(1, -1)

    # Standardize the input data (use the same scaler as in training)
    scaled_data = scaler.fit_transform(input_data)  # Ensure `scaler` matches the training scaler

    # Make prediction
    prediction = model.predict(scaled_data)
    predicted_class = np.argmax(prediction)

    # Display result
    if predicted_class == 0:
        st.error("The tumor is predicted to be Malignant (Cancerous).")
    else:
        st.success("The tumor is predicted to be Benign (Non-Cancerous).")

    st.write(f"Prediction Confidence: Malignant: {prediction[0][0]:.2f}, Benign: {prediction[0][1]:.2f}")
