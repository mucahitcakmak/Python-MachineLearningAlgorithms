# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("PROGRAM IS STARTING".center(50, "-"))
# DATA UPLOAD 
data = pd.read_csv("data/fullData.csv")


# DATA PREPROCESSİNG
# print(data)

# DATA IMPORT
age = data.iloc[:,1:4].values

from sklearn import preprocessing

# DATA LABEL ENCODING
from sklearn import preprocessing
country = data.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

# DATA LABEL ENCODING 2
gender = data.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
gender[:, -1] = le.fit_transform(data.iloc[:, -1])



# DATA MERGE
categoryResult = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "us"])
# print(categoryResult)
missingDataResult = pd.DataFrame(data=age, index=range(22), columns=["lenght", "weight", "age"])
# print(missingDataResult)
genderResult = pd.DataFrame(data=gender[:,:1], index=range(22), columns=["gender"])
# print(genderResult)

# tek tek değiştirip özelleştirdiğimiz listeleri
# birleştiriyoruz
s = pd.concat([categoryResult, missingDataResult], axis=1)
s2 = pd.concat([s, genderResult], axis=1)
# print(s)

# Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
# ölçülcek veriler , bağımlı değişken
x_train, x_test, y_train, y_test = train_test_split(s, genderResult, test_size=0.33, random_state=0)

# Verilerin ölçeklendirilmesi
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

lenght = s2.iloc[:,3:4].values
left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

dt = pd.concat([left,right], axis=1)

x_train, x_test, y_train, y_test = train_test_split(dt, lenght, test_size=0.33, random_state=0)
regression.fit(x_train, y_train)
y_pred = regression.predict(x_test)

print("PROGRAM IS FINISHED".center(50, "-"))