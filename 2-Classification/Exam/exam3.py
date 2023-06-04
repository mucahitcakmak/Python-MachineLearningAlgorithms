import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


data = pd.read_csv("data/exam3.csv")

# DataFrame
x = data.iloc[:,0:4] 
y = data.iloc[:,-1:]

# Numpy
Xn = x.values
Yn = y.values


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.33, random_state=0)


# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

print("Logistic Regression")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")



# KNN (K Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
knn.fit(X_train, y_train)
knnY_pred = knn.predict(X_test)

print("Knn")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")


# SVM
from sklearn.svm import SVC 
svc = SVC(kernel="linear") 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Svm")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")


# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Naive Bayes")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")



# DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

print("Decision Tree")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")



# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("Random Forest")
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")