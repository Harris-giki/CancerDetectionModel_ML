import pandas as pd #a data read/manipulating library
import seaborn as sns #a plotting library

data = pd.read_csv("data.csv") #import the dataset
data.head() #read the data
data.info() #gives the data of all the variables that we are going to have
data.describe() # can get the basic statistical features of the data
#cleaning the data
# 1. recognizing the field that are empty/have no information
sns.heatmap(data.isnull())
#dropping the column that has no data
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# we specify what we are deleting
data.head() # displaying the data from head

#converting the diagnosis data outputs into 1s and 0s outputs, as a part of the logistic regression model
data.diagnosis = (1 if value=="M" else 0 for value in data.diagnosis)
# now telling python to convert our data-type for diagnosis into category for logistic regression model
data["diagnosis"] = data["diagnosis"].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")

#dividing the dataset to target variables and predictors

y=data["diagnosis"] #target variable
x=data.drop("diagnosis", axis=1) #predictor variables
#Normalizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #create a scalar object

# fit the scalar to the data and transform the data

X_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X_scaled, y, test_size=0.30,random_state=42)

from sklearn.linear_model import LogisticRegression
# create the lr model
lr=LogisticRegression()

#train the model on the training data
lr.fit(X_train, y_train)

#predict the target variable based on test data
y_pred=lr.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))          