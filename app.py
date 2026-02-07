# Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model, metrics, svm, ensemble, preprocessing
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

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# universal method for model training

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # train model
    classifier.fit(feature_vector_train, label)
    
    # generate predictions for test set
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    # evaluate model
    scores = list(metrics.precision_recall_fscore_support(predictions, y_test))
    score_vals = [
        scores[0][0],
        scores[1][0],
        scores[2][0]
    ]
    score_vals.append(metrics.accuracy_score(predictions, y_test))
    return classifier, score_vals

# MODEL 1 - logistic regression
lr_model, accuracy = train_model(linear_model.LogisticRegression(), X_train, y_train, X_test)
accuracy_compare = {'LR': accuracy}
#print ("LR, accuracy: ", accuracy)

# MODEL 2 - Support Vector Machine
svm_model, accuracy = train_model(svm.SVC(probability=True), X_train, y_train, X_test)
accuracy_compare['SVM'] = accuracy
#print ("SVM, accuracy: ", accuracy)

# MODEL 3 - Random Forest Tree 
rft_model, accuracy = train_model(ensemble.RandomForestClassifier(), X_train, y_train, X_test)
accuracy_compare['RF'] = accuracy
#print ("RF, accuracy: ", accuracy)


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
prob_lr = lr_model.predict_proba(new_data)[:,1][0]
prob_svm = svm_model.predict_proba(new_data)[:,1][0]
prob_rf = rft_model.predict_proba(new_data)[:,1][0]

# Wyświetlenie wyników
st.subheader("Probability of passing an exams:")
st.write(f"Logistic Regression: {round(prob_lr*100, 2)}%")
st.write(f"Support Vector Machine: {round(prob_svm*100, 2)}%")
st.write(f"Random Forest: {round(prob_rf*100, 2)}%")