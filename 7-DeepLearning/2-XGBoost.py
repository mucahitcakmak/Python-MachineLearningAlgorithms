from math import pi
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# Hızlı çalışır
# Yüksek verilerde iyi performans gösterir


data = pd.read_csv("data/churnModeling.csv")

x = data.iloc[:, 3:13].values # Bağımsız
y = data.iloc[:, 13].values # Bağımlı



# Label Enconding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])

x[:,2] = le.fit_transform(x[:,2])


# One Hot Enconding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],remainder="passthrough")
x = ohe.fit_transform(x)
x = x[:,1:]


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)



# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



# XGBOOST
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print("XGBoost Cm:")
print(cm)

