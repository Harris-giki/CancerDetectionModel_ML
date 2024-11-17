import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Set up the Streamlit app
st.title("Breast Cancer Prediction App")
st.write("This application uses a Neural Network to predict if breast cancer is malignant or benign based on user inputs.")

# Load and process the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
X = data_frame.drop(columns=['label'], axis=1)
Y = data_frame['label']

# Calculate correlation with target label
correlation_matrix = X.corrwith(Y).abs()  # Absolute correlation to focus on magnitude
correlated_features = correlation_matrix[correlation_matrix > 0.1].index  # Select features with correlation > 0.1 (you can adjust this threshold)

# Filter the dataset to use only the selected correlated features
X = X[correlated_features]

# Display features and their correlations in a table
correlated_features_df = pd.DataFrame({
    'Required Features': correlated_features,
    'Correlation with Target Variable': correlation_matrix[correlated_features].values
})

st.header("Input Features")
st.write(correlated_features_df)  # Display the table with feature names and correlations

# Prepare data for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardizing the dataset
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Define and train the neural network
tf.random.set_seed(3)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train_std.shape[1],)),  # Dynamically use the number of selected features
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_std, Y_train, validation_split=0.1, epochs=15)

# User input for prediction
input_features = []
for feature in correlated_features:
    value = st.number_input(f"Enter value for {feature}:", format="%.4f", value=0.0)
    input_features.append(value)

# Predict button
if st.button("Predict"):
    # Convert input to numpy array and preprocess
    input_data_as_numpy_array = np.asarray(input_features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_std_data = scaler.transform(input_data_reshaped)

    # Make a prediction
    prediction = model.predict(input_std_data)
    prediction_label = np.argmax(prediction)

    # Display the result
    if prediction_label == 0:
        st.write("### The Breast Cancer is Malignant")
    else:
        st.write("### The Breast Cancer is Benign")

# Display training accuracy for reference
loss, accuracy = model.evaluate(X_test_std, Y_test)
st.write(f"Model Test Accuracy: {accuracy * 100:.2f}%")
