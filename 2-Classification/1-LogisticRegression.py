import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/fullData.csv")

x = data.iloc[:,1:4] # DataFrame | Bağımsız
y = data.iloc[:,-1:] # Bağımlı

Xn = x.values # Numpy array
Yn = y.values



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
# print(y_pred)
# print(y_test)



# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)