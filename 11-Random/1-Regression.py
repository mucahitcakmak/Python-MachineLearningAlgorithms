import pandas as pd
import numpy as np



# Data Importing
data = pd.read_csv("data/CompanyGain.csv")

x = data.iloc[:, -1:] # Bağımsız
y = data.iloc[:, :4] # Bağımlı 

country = data.iloc[:, 3:4].values




# Label Encoding and One Hot Encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
country[:, 0] = le.fit_transform(country[:, 0])
ohe = OneHotEncoder()
country = ohe.fit_transform(country).toarray()



# Data Merge
country = pd.DataFrame(data=country, index=range(50), columns=["ankara", "istanbul", "kocaeli"])
newData = pd.concat([y.iloc[:, :3], country], axis=1)




# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, newData, test_size=0.33, random_state=0)



# Multiple Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)


# OLS 
import statsmodels.api as sm
model = sm.OLS(y_pred, y_test)
print("Linear Reg".center(80, "-"))
print(model.fit().summary())


