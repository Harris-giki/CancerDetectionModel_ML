import pandas as pd  # For data manipulation
import seaborn as sns  # For visualization
import streamlit as st  # For creating the Streamlit app
from sklearn.preprocessing import StandardScaler  # For scaling the features
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import accuracy_score  # For evaluating model accuracy

# Step 1: Create the Streamlit interface
st.title("Breast Cancer Diagnosis Prediction")
st.write("### This model predicts whether a tumor is malignant or benign based on selected features.")

# Step 2: Load the dataset and perform initial data analysis
data = pd.read_csv("data.csv")  # Loading the dataset

# Dropping unnecessary columns and cleaning the data
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # Dropping columns with no data

# Step 3: Convert the target 'diagnosis' to numerical (1 = Malignant, 0 = Benign)
data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)

# Step 4: Perform correlation analysis to find the most important features
corr_matrix = data.corr()  # Calculate the correlation matrix
correlation_with_diagnosis = corr_matrix["diagnosis"].abs().sort_values(ascending=False)  # Get absolute correlation values

# Selecting top 10 features based on correlation with 'diagnosis'
top_features = correlation_with_diagnosis[1:11]  # Excluding the target 'diagnosis'
st.write("### Top 10 Features Correlated with Diagnosis", align="center")
st.write(top_features)

# Add a one-liner below the table
st.write("These features are considered most important in making accurate predictions through correlation statistical technique.")

# Step 5: Prepare the selected features
selected_features = top_features.index.tolist()  # Get the names of the top features
X = data[selected_features]  # Predictor variables
y = data["diagnosis"]  # Target variable

# Scaling the features to bring them to a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train the Logistic Regression model
lr = LogisticRegression()  # Instantiate the logistic regression model
lr.fit(X_scaled, y)  # Train the model on the data



# Input fields for the selected features
st.write("### Enter the feature values:")

input_data = []
for feature in selected_features:
    # Allow the user to input values for the selected features
    value = st.number_input(f"{feature}:", min_value=0.0, format="%.5f")
    input_data.append(value)

# Step 8: Predict and display results when the user clicks the 'Predict' button
if st.button("Predict Diagnosis"):
    # Convert input data into a DataFrame for prediction
    input_df = pd.DataFrame([input_data], columns=selected_features)
    
    # Scale the input data before making predictions
    input_scaled = scaler.transform(input_df)
    
    # Make the prediction
    prediction = lr.predict(input_scaled)
    prediction_prob = lr.predict_proba(input_scaled)[0][1]  # Probability of being malignant
    
    # Display the prediction result
    if prediction == 1:
        st.write("### The tumor is predicted to be **Malignant**.")
        st.write(f"Probability of being malignant: {prediction_prob:.2f}")
    else:
        st.write("### The tumor is predicted to be **Benign**.")
        st.write(f"Probability of being malignant: {prediction_prob:.2f}")

# Optional: Visualizing the distribution of the data
st.write("### Data Visualization")
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')  # Check for missing values visually
st.pyplot()

