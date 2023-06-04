import pandas as pd
import numpy as np

data = pd.read_csv("data/churnModeling.csv")

x = data.iloc[:,3:13] # Bağımsız değişken
y = data.iloc[:, -1:] # Bağımlı

Geography = data.iloc[:, 4:5].values
Gender = data.iloc[:, 5:6].values


# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Geography[:, 0] = le.fit_transform(data.iloc[:, 4])
Gender = le.fit_transform(data.iloc[:, 5])


# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
Geography = ohe.fit_transform(Geography).toarray()


# Data Merge
geoData = pd.DataFrame(data=Geography, index=range(10000), columns=["Fr", "Tr", "Us"])
genderData = pd.DataFrame(data=Gender, index=range(10000), columns=["Gender"])
remainingData = data.iloc[:, 6:13]

newData = pd.concat([geoData, genderData, remainingData], axis=1)
# print(newData)


# Train Test Split
Xn = newData.values
Yn = y.values.ravel()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.33, random_state=0)


# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

"""
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
"""

# Knn - K - Nearst Neighbor
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=4)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)


"""
# Confusuion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)
"""

# Cross Validation | Çapraz Doğrulama
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator=nn, X=X_train, y=y_train, cv=10)
print(cvs.mean())

"""
# Model Optimization
from sklearn.model_selection import GridSearchCV
p = [{"n_neighbors":[1,2,3,4,5]}]
gs = GridSearchCV(estimator=nn, param_grid=p, scoring="accuracy", cv=10, n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
bestResult = grid_search.best_score_
bestParameter = grid_search.best_params_
print(bestResult)
print(bestParameter)
"""