
<body>
    <h1>Breast Cancer Prediction Using Neural Networks</h1>
    <p>
        This project focuses on predicting breast cancer diagnosis (Malignant or Benign) using a neural network model 
        built with TensorFlow and Keras. Leveraging a dataset from the <strong>Scikit-learn library</strong>, this project 
        processes breast cancer data to train and evaluate a neural network for binary classification.
    </p>
    <h2>Objective</h2>
    <p>
        The primary objective of this project is to build a predictive system that accurately classifies breast cancer as 
        <strong>Malignant</strong> or <strong>Benign</strong> based on clinical features. This system could serve as a stepping stone for 
        more sophisticated medical diagnostic tools.
    </p>
    <h2>Dataset</h2>
    <ul>
        <li><strong>Features:</strong> 30 clinical attributes related to breast cancer, such as mean radius, texture, and symmetry.</li>
        <li><strong>Target Variable:</strong> 
            <ul>
                <li>0: Malignant (cancerous)</li>
                <li>1: Benign (non-cancerous)</li>
            </ul>
        </li>
        <li><strong>Size:</strong> 569 samples with no missing values.</li>
    </ul>
    <h2>Methodology</h2>
    <h3>1. Data Preprocessing</h3>
    <ul>
        <li>Loaded the dataset using Scikit-learn and converted it into a Pandas DataFrame for easier manipulation.</li>
        <li>Separated features (<code>X</code>) and target variable (<code>Y</code>).</li>
        <li>Standardized the features using <code>StandardScaler</code> to optimize model training.</li>
    </ul>
    <h3>2. Train-Test Split</h3>
    <p>The dataset was split into:</p>
    <ul>
        <li><strong>Training Set:</strong> 80% of the dataset.</li>
        <li><strong>Testing Set:</strong> 20% of the dataset.</li>
    </ul>
    <p>A <code>random_state</code> of 2 was used to ensure reproducibility.</p>  
    <h3>3. Neural Network Architecture</h3>
    <ul>
        <li><strong>Input Layer:</strong> Flattened input with 30 features to convert the matrix into a 1D array.</li>
        <li><strong>Hidden Layer:</strong> 20 neurons with ReLU activation.</li>
        <li><strong>Output Layer:</strong> 2 neurons with Sigmoid activation for binary classification.</li>
    </ul>   
    <h3>4. Model Training</h3>
    <ul>
        <li><strong>Loss Function:</strong> Sparse Categorical Crossentropy.</li>
        <li><strong>Optimizer:</strong> Adam.</li>
        <li><strong>Metric:</strong> Accuracy.</li>
        <li><strong>Epochs:</strong> 15 iterations of training over the dataset.</li>
    </ul>
    <h3>5. Evaluation</h3>
    <p>Visualized accuracy and validation accuracy trends and computed test accuracy and loss.</p>  
    <h3>6. Prediction System</h3>
    <p>A predictive system was implemented to classify new input data:</p>
    <ul>
        <li>Input data is reshaped and standardized.</li>
        <li>Predictions are generated using the trained model.</li>
        <li>Predicted class is determined using the <code>argmax</code> function.</li>
    </ul> 
    <h2>Results</h2>
    <p>
        The model achieved a good balance of accuracy on both the training and validation sets, indicating that it is neither 
        underfitting nor overfitting. Test accuracy was also satisfactory, demonstrating effective generalization.
    </p>
    <h2>Key Concepts</h2>
    <ul>
        <li>Data Preprocessing: Cleaning and normalizing data to optimize model performance.</li>
        <li>Neural Networks: Understanding layers, activations, and optimization techniques.</li>
        <li>Binary Classification: Using categorical cross-entropy and sigmoid activation.</li>
        <li>Standardization: Ensuring feature values are on the same scale for effective training.</li>
    </ul>  
    <h2>Future Enhancements</h2>
    <ul>
        <li>Experiment with deeper architectures or different activation functions.</li>
        <li>Incorporate additional datasets for a more robust model.</li>
        <li>Investigate feature importance and reduce dimensionality.</li>
        <li>Deploy the model as a web application using Flask/Django for real-time predictions.</li>
        <li>Use tools like SHAP or LIME for explainability to enhance trust in the model.</li>
        <li>Validate predictions on real-world patient data through healthcare collaboration.</li>
    </ul>
    <h2>Folder Structure</h2>
    <ul>
        <li><strong>BreastCancerPrediction_NeuralNetworks.ipynb:</strong> Main Jupyter Notebook with the complete code.</li>
        <li><strong>README.md:</strong> Documentation of the project.</li>
        <li><strong>requirements.txt:</strong> Dependencies (e.g., TensorFlow, Scikit-learn, Pandas, Matplotlib).</li>
    </ul>
        <h2>Prerequisites</h2>
    <ul>
        <li>Python 3.x</li>
        <li>Libraries: <code>numpy</code>, <code>pandas</code>, <code>matplotlib</code>, <code>tensorflow</code>, <code>sklearn</code></li>
    </ul>
    <h2>Conclusion</h2>
    <p>
        This project demonstrates how neural networks can be applied to medical data for predictive analysis. By focusing on 
        preprocessing, architecture design, and evaluation, it provides a foundational framework for implementing machine 
        learning in healthcare. With further development and validation, it could contribute to early and accurate diagnosis 
        of breast cancer.
    </p>
    <h2>Author</h2>
    <p><strong>Muhammad Haris</strong></p>
    <p>
        <strong>Email:</strong> <a href="mailto:harris.giki@gmail.com">harris.giki@gmail.com</a><br>
    </p>
</body>
</html>
