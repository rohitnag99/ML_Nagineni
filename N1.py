#Importing required libraries
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

#Intialize streamlit title for header and main body content
st.title("EE 658/758 - Assignment #5")
st.write("by Rohit Nagineni")
st.write("This tool helps you to predict output by using different ML models....Please enter the inputs in the sidebar.")

#Loading dataset by getting input from frontend 
df_name = st.sidebar.selectbox("Select Dataset", ("IRIS", "Digits"))
if df_name == "IRIS":
    df = datasets.load_iris()
elif df_name == "Digits":
    df = datasets.load_digits()

#Intialize required ML classifiers in a dictionary
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Neural Networks": MLPClassifier(max_iter=1000),
    "Naïve Bayes": GaussianNB(),
    "Support Vector Machine": SVC(C=1.0, kernel='rbf', gamma='scale'),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state = 0) #extra  model
}

#Select required classifer from frontend 
classifier_name = st.sidebar.selectbox("Select ML classifier", list(classifiers.keys()))

#Load feature names into a variable
features = df.feature_names

#Load train and test data for ML modeling
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scalling the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Take features input from the frontend UI
st.sidebar.title("Select the feature values")
feature_list = []
for i, feature in enumerate(features):
    f = st.sidebar.number_input(f"{feature}",max_value=1000.0,min_value=0.0,value=1.00, step=1.0)
    feature_list.append(f)

#Train the ML model
inputs = np.array(feature_list).reshape(1, -1)
model = classifiers[classifier_name]
model.fit(X_train, y_train)

#Predict the output for feature inputs by the user
prediction = model.predict(inputs)

# Plot Accuracy graph
a = []
c_names = []
for clf_name, clf_model in classifiers.items():
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    a.append(acc)
    c_names.append(clf_name)

# Sort accuracies and classifier names
s_data = sorted(zip(a, c_names), reverse=False)
s_acc, s_c_names = zip(*s_data)

# Plot sorted Accuracy graph
st.subheader("Accuracy Comparison ")
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(s_c_names, s_acc, color='skyblue')
ax.set_xlabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
st.pyplot(fig)

#Disaplay predicted vales in the frontend
st.subheader("Prediction")
if df_name == "IRIS":
    st.write("For classier <span style='color:#76ABAE;'> ", classifier_name,"</span>", "the predicted Iris Species is:  <span style='color:#F5E8C7;'> ", df.target_names[prediction[0]], "</span>", unsafe_allow_html=True)
else:
    st.write("For classier <span style='color:#76ABAE;'> ", classifier_name,"</span>", "the predicted Digit is:<span style='color:#F5E8C7;'>", prediction[0] ,"</span>", unsafe_allow_html=True)

#Display Accuracy in the frontend in sidebar
st.sidebar.subheader("Model Accuracy")
y_pred = model.predict(X_test)
accu = accuracy_score(y_test, y_pred)
st.sidebar.write("Accuracy:", round(accu*100,2))

#Display Confusion Matrix in the frontend
try:
    confusion = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    st.write(confusion)
except:
    st.write("Unable to generate confusion matrix.")




