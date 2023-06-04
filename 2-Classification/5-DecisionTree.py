import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix # CM

# Decision Tree ile sınıflandırma

data = pd.read_csv("data/fullData.csv")

x = data.iloc[:,1:4] # Bağımsız | DataFrame
y = data.iloc[:,-1:] # Bağımlı

Xn = x.values # Numpy
Yn = y.values


# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.33, random_state=0)


# Standart Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy") # Defalut Criterin = "gini"
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred) # sol üst, sağ alt = doğru sayısı
print(cm)

