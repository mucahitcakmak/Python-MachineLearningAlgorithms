import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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



from sklearn.svm import SVC # svm = Supported Vector Machine
svc = SVC(kernel="linear") # rbf best | aralarındaki
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred) # sol üst sağ alt doğruluk payı

print(cm)
