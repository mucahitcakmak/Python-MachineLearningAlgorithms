import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = pd.read_csv("data/fullData.csv")

x = data.iloc[:,1:4] # DataFrame | Bağımsız
y = data.iloc[:,-1:] # Bağımlı

Xn = x.values # Numpy array
Yn = y.values


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn,test_size=0.33,random_state=0)

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# SVC
from sklearn.svm import SVC # svm = Supported Vector Machine
svc = SVC(kernel="rbf") # rbf best | aralarındaki (her zaman değil)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred) # sol üst sağ alt doğruluk payı
print(cm)



# K-Fold Cross Validation | Çapraz Doğrulama
from sklearn.model_selection import cross_val_score
"""
1 - estimator = hangi algoritma (Linear, SVM, KNN..) 
2 - X
3 - Y
4 - cv = kaça katlamalı
"""
cvs = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=4)
print(cvs.mean()) # ne kadar yüksekse iyi
print(cvs.std()) # ne kadar düşükse iyi