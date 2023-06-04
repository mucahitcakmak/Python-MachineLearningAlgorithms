import pandas as pd
import numpy as np
from matplotlib import colors, pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# Y = a + bX | X = bağımlı değişken, Y = bağımsız değişken

df = pd.read_csv("data/example1.csv", sep=";")

# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.scatter(df.alan, df.fiyat, color="red", marker="+")
# plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.show()

# choise = input("Kaç Metrekare: ")

# Linear Regression Model

reg = LinearRegression()
# Tek köşeli parantez Series verir | İki köşeli parantez dataframe verir
reg.fit(df[["alan"]], df["fiyat"]) # önce x sonra y değeri verilir | fit(x, y) | fit(bağımsız, bağımlı) 
result = reg.predict([[275]]) # 2D değer döndürmek için paranteze aldık | 2d array oldu parantezler sayesinde


plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.alan, df.fiyat, color="red", marker="+")
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.plot(df.alan, reg.predict(df[["alan"]]), color="blue")
plt.show()



"""
print(type(df.iloc[:,1])) # Series - 1D
print(type(df.iloc[:,:1])) # DataFrame - 2D | alan ve fiyat kolonlarını alır
print(type(df["alan"])) # Series - 1D
print(type(df[["alan"]])) # DataFrame - 2D
"""


"""
# Y = a + bX | b değerini bulur | b değeri çizginin eğimini verir
result = reg.coef_
# print(result)

# Y = a + bX | a değerini bulur | a sabit değerdir
result = reg.intercept_
# print(result)

# Sağlama yapıyoruz 
a = reg.intercept_
b = reg.coef_
x = 275
y = a + b * x
print(y)
"""


