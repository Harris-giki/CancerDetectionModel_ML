import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set up the main title
st.title("Breast Cancer Prediction Model")
st.write("Hey there! Today, we’ll be working through a Logistic Regression model to classify tumor cells. Let's dive in!")

# Step 1: Load the data
st.header("1. Data Overview")
st.write("First things first, let's load up the data and take a peek at its structure.")

# Uploading the dataset file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Here are the first few rows of our dataset:")
    st.write(data.head())

    # Display basic information
    st.subheader("Dataset Info")
    buffer = st.empty()  # For holding the buffer output of data info
    with st.expander("Expand to see dataset information"):
        data_info_buffer = data.info(buf=buffer)
        st.text(buffer)

    # Display statistical summary
    st.subheader("Statistical Summary")
    st.write(data.describe())

    # Step 2: Visualizing Missing Data
    st.header("2. Data Cleaning")
    st.write("Now, let's check if there are any missing values in the dataset. I'll display a heatmap to show missing entries.")
    fig, ax = plt.subplots()
    sns.heatmap(data.isnull(), ax=ax, cbar=False)
    st.pyplot(fig)

    # Dropping unnecessary columns
    st.write("We'll remove columns with no data or irrelevant information for our model.")
    if "Unnamed: 32" in data.columns:
        data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    st.write("Updated data:")
    st.write(data.head())

    # Step 3: Encoding the Target Variable
    st.header("3. Encoding the Target Variable")
    st.write("To simplify the model's job, let’s convert the `diagnosis` variable into binary form (Malignant = 1, Benign = 0).")

    # Encoding the diagnosis column
    data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    data["diagnosis"] = data["diagnosis"].astype("category", copy=False)
    st.write("Here's the count of each class after encoding:")
    st.bar_chart(data["diagnosis"].value_counts())

    # Step 4: Splitting the Data
    st.header("4. Preparing the Data for Modeling")
    st.write("Now, let's separate our target variable (`diagnosis`) from our features and standardize the feature data to prepare it for modeling.")

    # Defining predictors and target
    y = data["diagnosis"]
    X = data.drop("diagnosis", axis=1)

    # Normalizing the predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    st.write("Data has been split into training and test sets.")

    # Step 5: Training the Model
    st.header("5. Training the Model")
    st.write("Let’s fit a logistic regression model to our data and see how well it performs.")

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    st.write("The model has been successfully trained!")

    # Step 6: Making Predictions and Evaluating
    st.header("6. Making Predictions and Evaluating the Model")
    st.write("Let’s use our test data to make predictions and evaluate the model's accuracy.")

    # Predictions
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Display classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("And there we have it! We've successfully built, trained, and evaluated our model on breast cancer data. 🚀")
