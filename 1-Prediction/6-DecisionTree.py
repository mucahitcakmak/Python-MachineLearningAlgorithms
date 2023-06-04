import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("data/salarys.csv")


x = df.iloc[:,1:2] # DataFrame
y = df.iloc[:,2:]
Xn = x.values # Numpy array
Yn = y.values


# DECISION TREE
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0) #
dtr.fit(Xn, Yn)

plt.scatter(Xn, Yn)
plt.plot(Xn, dtr.predict(Xn))
plt.show()


# Yuvarlama gibi birş ey yapıyor
# Maaş lardan birini seçiyo asla ortasını vs almıyor
print(dtr.predict([[11]]))
print(dtr.predict([[6.6]]))
print(dtr.predict([[6.4]]))


