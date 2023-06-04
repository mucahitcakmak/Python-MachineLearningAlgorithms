import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor # Decision Tree
from sklearn.metrics import r2_score # R-Square

df = pd.read_csv("data/salarys.csv")


x = df.iloc[:,1:2] # DataFrame
y = df.iloc[:,2:]
Xn = x.values # Numpy array
Yn = y.values




# DECISION TREE
dtr = DecisionTreeRegressor(random_state=0) #
dtr.fit(Xn, Yn)

# Traininglerde başarılıdır ama gerçekde öyle değildir
# Herşeyin üzerinden geçtiği için
print("Decision Tree - R2".center(50, "-"))
print(r2_score(Yn, dtr.predict(Xn)))




# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators = kaç tane karar ağacı çizsin
rfr.fit(Xn, Yn)



# Random Forest R-SQUARE
# 1 e ne kadar yakınsa o kadar doğruluk payı vardır
# sayı eksilerdeyse beterin beteridir
print("Random Forest - R2".center(50, "-"))
print(r2_score(Yn, rfr.predict(Xn)))



