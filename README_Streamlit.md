   <h1>Breast Cancer Diagnosis Prediction Using Logistic Regression</h1>
    <p>This Streamlit application predicts whether a breast tumor is <strong>malignant</strong> or <strong>benign</strong> based on a set of selected features derived from a dataset of breast cancer diagnostic data. The model uses <strong>Logistic Regression</strong> as the machine learning algorithm and leverages <strong>correlation analysis</strong> to select the most relevant features for prediction.</p>
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#project-approach">Project Approach</a></li>
        <li><a href="#installation-instructions">Installation Instructions</a></li>
        <li><a href="#how-to-use">How to Use</a></li>
        <li><a href="#code-explanation">Code Explanation</a></li>
        <li><a href="#future-directions">Future Directions</a></li>
        <li><a href="#license">License</a></li>
    </ul>
    <h2 id="overview">Overview</h2>
    <p>Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early detection is crucial for improving survival rates. This project provides a simple interface where users can input selected features of breast cancer data and receive a prediction on whether the tumor is malignant or benign.</p>
    <p>The model uses <strong>Logistic Regression</strong>, a statistical method, and performs <strong>correlation analysis</strong> to identify the top features most relevant to the diagnosis. The application is built using <strong>Streamlit</strong>, a popular framework for creating data-driven applications with minimal effort.</p>
    <h2 id="project-approach">Project Approach</h2>
    <ol>
        <li><strong>Data Preprocessing</strong>:
            <ul>
                <li><strong>Data Loading</strong>: The dataset is loaded into the application from a CSV file.</li>
                <li><strong>Feature Selection</strong>: We performed a <strong>correlation analysis</strong> to select the top 10 most important features based on their correlation with the target variable (diagnosis).</li>
                <li><strong>Data Cleaning</strong>: Unnecessary columns (like <code>Unnamed: 32</code> and <code>id</code>) were dropped.</li>
                <li><strong>Data Conversion</strong>: The diagnosis column was converted into a binary format (1 for malignant and 0 for benign).</li>
            </ul>
        </li>
        <li><strong>Machine Learning Model</strong>:
            <ul>
                <li>We used <strong>Logistic Regression</strong> to train a predictive model.</li>
                <li>The model is trained on the selected features of the dataset, and the user can make predictions by entering new data points.</li>
            </ul>
        </li>
        <li><strong>Streamlit Interface</strong>:
            <ul>
                <li>A user-friendly interface is created where users can input values for the selected features, and the model will predict whether the tumor is benign or malignant.</li>
                <li>The application shows the correlation-based selected features in a table and explains their significance in making accurate predictions.</li>
            </ul>
        </li>
    </ol>
    <h2 id="installation-instructions">Installation Instructions</h2>
    <p>To run this project locally, follow these steps:</p>
    <h3>1. Clone the repository:</h3>
    <pre><code>git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction</code></pre>
    <h3>2. Create a virtual environment (optional but recommended):</h3>
    <pre><code>python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`</code></pre>   
    <h3>3. Install the required dependencies:</h3>
    <pre><code>pip install -r requirements.txt</code></pre>    
    <p><strong>requirements.txt</strong> should include:</p>
    <pre><code>pandas
seaborn
scikit-learn
streamlit</code></pre>
    <h3>4. Run the Streamlit app:</h3>
    <pre><code>streamlit run app.py</code></pre>
    <p>This will launch the app in your browser.</p>
    <h2 id="how-to-use">How to Use</h2>
    <ol>
        <li><strong>Input Feature Values</strong>: You will see fields for the top 10 most important features of the dataset.</li>
        <li><strong>Enter values</strong> for each feature using the input fields.</li>
        <li><strong>Click 'Predict'</strong>: After entering the values, click on the <strong>'Predict'</strong> button to get the diagnosis prediction (benign or malignant).</li>
        <li><strong>View Results</strong>: The model will display whether the tumor is predicted to be <strong>malignant</strong> or <strong>benign</strong> along with the probability of being malignant.</li>
    </ol>
    <h2 id="code-explanation">Code Explanation</h2>
    <h3>Data Loading and Cleaning</h3>
    <pre><code>data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # Dropping unnecessary columns
data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)  # Convert 'M' to 1 (malignant) and 'B' to 0 (benign)</code></pre>
    <h3>Correlation Analysis</h3>
    <p>We compute the correlation of all features with the target variable (<code>diagnosis</code>) to identify the most important ones:</p>
    <pre><code>corr_matrix = data.corr()
correlation_with_diagnosis = corr_matrix["diagnosis"].abs().sort_values(ascending=False)
top_features = correlation_with_diagnosis[1:11]</code></pre>
    <h3>Model Training</h3>
    <pre><code>scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegression()
lr.fit(X_scaled, y)</code></pre>
    <h3>User Interface with Streamlit</h3>
    <p>The user interface allows the user to input data, and when they click the <strong>'Predict'</strong> button, the model will make a prediction based on the entered data:</p>
    <pre><code>if st.button("Predict Diagnosis"):
    input_df = pd.DataFrame([input_data], columns=selected_features)
    input_scaled = scaler.transform(input_df)
    prediction = lr.predict(input_scaled)
    prediction_prob = lr.predict_proba(input_scaled)[0][1]</code></pre>
    <h2 id="future-directions">Future Directions</h2>
    <ul>
        <li><strong>Model Improvement</strong>: We can experiment with other machine learning algorithms (like Random Forest or Support Vector Machines) to improve prediction accuracy.</li>
        <li><strong>Additional Features</strong>: Integrating more advanced statistical methods for feature selection and performance tuning.</li>
        <li><strong>User Experience</strong>: Improve the user interface by providing visual feedback on input data and model predictions.</li>
        <li><strong>Web Deployment</strong>: Deploy the Streamlit app on a cloud platform like Heroku or AWS for easier access.</li>
        <li><strong>Real-time Prediction</strong>: Integrate the model with real-time data sources for predictions based on incoming patient data.</li>
    </ul>

