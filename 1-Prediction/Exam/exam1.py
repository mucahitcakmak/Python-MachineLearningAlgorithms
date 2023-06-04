import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import statsmodels.api as sm


data = pd.read_csv("data/exam1.csv")
outlook = data[["outlook"]].values
windy = data[["windy"]].values
play = data[["play"]].values

# LABEL ENCODING
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

encodingData = data.apply(le.fit_transform)
"""
outlook[:, 0] = le.fit_transform(outlook[:, 0]) # Array den Series oldu [:, 0]
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

windy[:, 0] = le.fit_transform(windy[:, 0])
windy = ohe.fit_transform(windy).toarray()
print(windy)

play[:, 0] = le.fit_transform(play[:, 0])
print(play)
"""

oheData = encodingData[["outlook"]]
oheData = ohe.fit_transform(oheData).toarray()

oheDataFrame = pd.DataFrame(data=oheData, index=range(14), columns=["overcast", "rainy", "sunny"])
data1 = pd.concat([encodingData.iloc[:,3:], data.iloc[:,1:3]], axis=1)
newData = pd.concat([oheDataFrame, data1], axis=1)

dependent = newData.iloc[:,:5]
independent = newData.iloc[:,-1:]

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = model_selection.train_test_split(dependent, independent, test_size=0.33, random_state=0)


# DATA SCALING
re = linear_model.LinearRegression()
re.fit(x_train, y_train)
y_pred = re.predict(x_test)

# print(y_test)
# print(y_pred)


# BECKWARD ELIMINATION
x = np.append(arr=np.ones((14,1)).astype(int), values=newData.iloc[:,:-1], axis=1)

x_l = newData.iloc[:,[0,1,2,3,4,5]]
x_l = np.array(x_l, dtype=float)
model = sm.OLS(independent, x_l).fit()
# print(model.summary())


x_l = newData.iloc[:,[0,1,2,4,5]]
x_l = np.array(x_l, dtype=float)
model = sm.OLS(independent, x_l).fit()
# print(model.summary())

# Windy çıkartıyoruz | tahmin olasılığını düşürdüğü için
del x_train["windy"]
del x_test["windy"]

# Tekrar eğit ve tahmin et
re.fit(x_train, y_train)
y_pred = re.predict(x_test)
print(y_pred)
print(y_test) # tahmin etmek istenilen

