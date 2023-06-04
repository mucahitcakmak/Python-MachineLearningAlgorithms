import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Linear Reg
from sklearn.preprocessing import PolynomialFeatures # Polynomial Reg
import statsmodels.api as sm # P value

data = pd.read_csv("data/exam2.csv")

rankLevel = data.iloc[:,2:5] # Bağımsız değişkenler | DataFrame
salary = data.iloc[:,-1:]

rankLevelN = rankLevel.values # Numpy array
salaryN = salary.values




# Linear Reg
lr = LinearRegression()
lr.fit(rankLevelN, salaryN)

# P value
model = sm.OLS(lr.predict(rankLevelN), rankLevelN)
print("Linear Reg".center(80, "-"))
print(model.fit().summary())





# Polynomial Reg
pf = PolynomialFeatures(degree=2) # 2. dereceden obje oluştur
x_poly = pf.fit_transform(rankLevelN)

lr2 = LinearRegression()
lr2.fit(x_poly, salaryN)

model2 = sm.OLS(lr2.predict(pf.fit_transform(rankLevelN)), rankLevelN)
print("Polynomial Reg".center(80, "-"))
print(model2.fit().summary())






# SVR
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scale = sc1.fit_transform(rankLevelN)
sc2 = StandardScaler()
y_scale = np.ravel(sc2.fit_transform(salaryN.reshape(-1,1)))

from sklearn.svm import SVR
svr = SVR(kernel="rbf")
svr.fit(x_scale, y_scale)

model3 = sm.OLS(svr.predict(x_scale), x_scale)
print("SVR Reg".center(80, "-"))
print(model3.fit().summary())






# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0) 
dtr.fit(rankLevelN, salaryN)

model4 = sm.OLS(dtr.predict(rankLevelN), rankLevelN)
print("Decision Tree".center(80, "-"))
print(model4.fit().summary())





# Random Forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators = kaç tane karar ağacı çizsin
rfr.fit(rankLevelN, salaryN)

model5 = sm.OLS(rfr.predict(rankLevelN), rankLevelN)
print("Random Forest".center(80, "-"))
print(model5.fit().summary())