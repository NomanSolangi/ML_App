import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

st.write(""" 
         # Explore different machine learning models on different datasets
         ## Let see which is the best.""")

#Add data sets in the side bar
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Breast Cancer", "Digits"))

#Add model in the side bar
model_name = st.sidebar.selectbox("Select Model", ("KNN", "SVM", "Random Forest")) 

#Load dataset

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    x = data.data
    y = data.target
    return x, y

X, y = get_dataset(dataset_name)
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

def add_parameter_ui(model_name):
    params = dict()
    if model_name == "SVM":
        C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
        params['C'] = C
    elif model_name == "KNN":
        K = st.sidebar.slider("K (Number of neighbors)", 1, 15, 5)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15, 5)
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 100, 50)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        
    return params

params = add_parameter_ui(model_name)

def get_model(model_name, params):
    
    clf = None
    if model_name == "SVM":
        clf = SVC(C=params['C'])
    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
    return clf

clf = get_model(model_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model: {model_name}")   
st.write(f"Accuracy: {accuracy:.2f}")

#PCA

pca = PCA(2)
X_pca = pca.fit_transform(X)

x1 = X_pca[:, 0]
x2 = X_pca[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

