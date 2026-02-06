# Imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model, metrics, svm, ensemble
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("exams.csv")

        # Deleting not usefull data
df = df.drop('race/ethnicity', axis=1)

educationLvl = df
educationLvl = educationLvl.drop('lunch', axis=1)
educationLvl = educationLvl.drop('gender', axis=1)
educationLvl = educationLvl.drop('test preparation course', axis=1)
educationLvl = educationLvl.groupby("parental level of education").mean()

df["associate's degree"] = 0
df["bachelor's degree"] = 0
df["high school"] = 0
df["master's degree"] = 0
df["some college"] = 0
df["some high school"] = 0

lunch = df[["lunch"]]
lunch = lunch.groupby("lunch").mean()
df["had lunch"] = 0

test_prep = df[["test preparation course"]]
test_prep = test_prep.groupby("test preparation course").mean()

df["completed preparation course"] = 0

df["male"] = 0
df["female"] = 0

# one-hot encoding for genders
genders = [
    "male",
    "female",
]
for gend in genders:
    df[gend] = (df["gender"] == gend).astype(int)
df = df.drop("gender", axis=1)

# binary encoding for those who finished course
df["completed preparation course"] = (df["test preparation course"] == "completed").astype(int)
df = df.drop("test preparation course", axis=1)

# binary encoding for those who ate lunch
df["had lunch"] = (df["lunch"] == "standard").astype(int)
df = df.drop("lunch", axis=1)

# one-hot encoding for parental level of education
levels = [
    "associate's degree",
    "bachelor's degree",
    "high school",
    "master's degree",
    "some college",
    "some high school",
]

for lvl in levels:
    df[lvl] = (df["parental level of education"] == lvl).astype(int)
df = df.drop("parental level of education", axis=1)

# average score of 3 exams
df["avg score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# passed exam if scores of all exams are higher than 60
df['passed exam'] = ((df['math score'] > 60) & 
                     (df['reading score'] > 60) & 
                     (df['writing score'] > 60)).astype(int)

# create training set
categories = [
    "associate's degree", "bachelor's degree", "high school", "master's degree",
    "some college", "some high school", "had lunch", "completed preparation course",
    "male", "female"
]
X = df[categories]

# create labels for training set
#y = df.loc[:, 'passed exam'].to_numpy()
y = df["passed exam"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345, stratify=y,)

# universal method for training and evaluation of models

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # train model
    classifier.fit(feature_vector_train, label)
    # universal method for training and evaluation of models

def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # train model
    classifier.fit(feature_vector_train, label)

    # generate labels for validation set
    predictions = classifier.predict(feature_vector_valid)

    # evealuate model based on test set labels
    scores = list(metrics.precision_recall_fscore_support(predictions, y_test))
    score_vals = [
        scores[0][0],
        scores[1][0],
        scores[2][0]
    ]
    score_vals.append(metrics.accuracy_score(predictions, y_test))
    return score_vals
    # generate labels for validation set
    predictions = classifier.predict(feature_vector_valid)

    # evealuate model based on test set labels
    scores = list(metrics.precision_recall_fscore_support(predictions, y_test))
    score_vals = [
        scores[0][0],
        scores[1][0],
        scores[2][0]
    ]
    score_vals.append(metrics.accuracy_score(predictions, y_test))
    return score_vals


unique, counts = np.unique(y, return_counts=True)
#print(dict(zip(unique, counts)))



# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
accuracy_lr = accuracy_score(y_test, lr_pipeline.predict(X_test))
#print("LR Accuracy:", accuracy_lr)

# SVM pipeline
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", svm.SVC(class_weight="balanced", probability=True)) 
])
svm_pipeline.fit(X_train, y_train)
accuracy_svm = accuracy_score(y_test, svm_pipeline.predict(X_test))
#print("SVM Accuracy:", accuracy_svm)

rf = RandomForestClassifier(
    n_estimators=200, 
    random_state=13579,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
accuracy_rf = accuracy_score(y_train, y_train)
#print("Random Forest Test Accuracy:", accuracy_rf)
df["passed exam"].value_counts(normalize=True)


import streamlit as st

# ui
st.title("Prediction of passing an exams: math, reading, writing.")

gender = st.selectbox("Gender", ["male", "female"])
course = st.selectbox("Completed course?", ["Yes", "No"])
lunch = st.selectbox("Ate lunch?", ["Yes", "No"])
education = st.selectbox("Completed education", [
    "associate's degree", "bachelor's degree", "high school", "master's degree",
    "some college", "some high school"
])

# Konwersja inputu na format datafreame
new_data = pd.DataFrame([{col:0 for col in categories}])

# Ustawienie odpowiednich kolumn na 1
new_data[gender] = 1
new_data["had lunch"] = 1 if lunch=="Yes" else 0
new_data["completed preparation course"] = 1 if course=="Yes" else 0
new_data[education] = 1

# prawdopodobienstwo
prob_lr = lr_pipeline.predict_proba(new_data)[:,1][0]
prob_svm = svm_pipeline.predict_proba(new_data)[:,1][0]
prob_rf = rf.predict_proba(new_data)[:,1][0]

# Wyświetlenie wyników
st.subheader("Probability of passing an exams:")
st.write(f"Logistic Regression: {round(prob_lr*100, 2)}%")
st.write(f"SVM: {round(prob_svm*100, 2)}%")
st.write(f"Random Forest: {round(prob_rf*100, 2)}%")