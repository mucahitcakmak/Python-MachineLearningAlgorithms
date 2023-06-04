import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("data/salarys.csv")


x = df.iloc[:,1:2] # DataFrame
y = df.iloc[:,2:]
Xn = x.values # Numpy array
Yn = y.values


# RANDOM FOREST
# ensemble = birden fazla kişiden oluşan bir grup
# Veriyi parçalara bölüp birden çok karar ağacı(decision tree) oluşturuyo
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators = kaç tane karar ağacı çizsin
rfr.fit(Xn, Yn)

plt.scatter(Xn, Yn, color="red")
plt.plot(Xn, rfr.predict(Xn))
plt.show()


print(rfr.predict([[11]]))
print(rfr.predict([[6.6]]))

