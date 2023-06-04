import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("data/salarys.csv")


x = df.iloc[:,1:2] # DataFrame
y = df.iloc[:,2:]
Xn = x.values # Numpy array
Yn = y.values



# LINEAR MODEL
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xn, Yn)


# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2) # 2. dereceden obje oluştur
x_poly = pf.fit_transform(Xn)

lr2 = LinearRegression()
lr2.fit(x_poly, y)


# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pf2 = PolynomialFeatures(degree=4) # derece arttıkça daha yakın sonuçlar gelir
x_poly2 = pf2.fit_transform(Xn)

lr3 = LinearRegression()
lr3.fit(x_poly2, y)



# Görselleştirmeler
plt.scatter(Xn, Yn, marker="D", color="red")

# Linear
plt.plot(x, lr.predict(Xn), color="blue")

# Polynomial 2. dereceden
plt.plot(Xn, lr2.predict(x_poly), color="yellow")

# Polynomial 4. dereceden
plt.plot(Xn, lr3.predict(x_poly2), color="green")

plt.show()


"""
# Tahminler
# Education Level = bağımsız değişkendi o yüzden tahmin etmek istediğimiz eğitim seviyesi
# Linear Regression tahminleri
print("Linear Regression Predict".center(50, "-"))
print(lr.predict([[11]])) 
print(lr.predict([[6.6]]))

# Polynomial Regression tahminleri | 2. dereceden
print("Polynomial Regression Predict".center(50, "-"))
print(lr2.predict(pf.fit_transform([[11]])))
print(lr2.predict(pf.fit_transform([[6.6]])))

# Polynomial Regression tahminleri | 4. dereceden
print("Polynomial Regression Predict".center(50, "-"))
print(lr3.predict(pf2.fit_transform([[11]])))
print(lr3.predict(pf2.fit_transform([[6.6]])))
"""



# R2
print("Linear Reg - R2".center(50, "-"))
print(r2_score(Yn, lr.predict(Xn)))
print("Polynomial Reg- degree 2 - R2".center(50, "-"))
print(r2_score(Yn, lr2.predict(pf.fit_transform(Xn))))
print("Polynomial Reg- degree 4 - R2".center(50, "-"))
print(r2_score(Yn, lr3.predict(pf2.fit_transform(Xn))))

