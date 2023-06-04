import pandas as pd
import numpy as np


data = pd.read_csv("data/metin2Sales.csv")

x = data.iloc[:, 0:1] # Bağımsız
y = data.iloc[:, 1:2]




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
# print(y_pred)


# OLS 
import statsmodels.api as sm
model = sm.OLS(y_pred, y_test)
print("Linear Reg".center(80, "-"))
print(model.fit().summary())