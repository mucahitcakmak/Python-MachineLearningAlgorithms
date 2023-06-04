# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# DATA UPLOAD 
data = pd.read_csv("data/prediction1.csv")


# DATA PREPROCESSİNG *****************

# DATA IMPORT
# month = data.iloc[:,:-1].values
# sales = data.iloc[:,1].values
month = data[["Month"]]
sales = data[["Sales"]]


# Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
# (bağımsız değişken, bağımlı değişken, bölünme oranı, rastgelelik(seed gibi))
x_train, x_test, y_train, y_test = train_test_split(month, sales, test_size=0.33, random_state=0) 



"""
# Verilerin ölçeklendirilmesi | veriler oranları korunarak indirgeniyo
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

# Model İnşası | Linear Regression oluşumu
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) # neyi neye oranlamak istiyosun
prediction = lr.predict(x_test)

print(lr.predict([[10]]))

"""
# Veri Görselleştirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()
"""


