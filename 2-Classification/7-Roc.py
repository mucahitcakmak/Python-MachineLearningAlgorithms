import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


data = pd.read_csv("data/fullData.csv")

x = data.iloc[:,1:4] # Bağımsız | DataFrame
y = data.iloc[:,-1:] # Bağımlı 

Xn = x.values # Numpy
Yn = y.values


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.33, random_state=0)


# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)



# Roc
y_proba = rfc.predict(X_test)
print("Real True", y_test)
print("Roc:", y_proba[:,0])


from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label="e")
print(fpr, tpr, thold)

