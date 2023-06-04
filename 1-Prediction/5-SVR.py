import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score # R-Square


df = pd.read_csv("data/salarys.csv")


x = df.iloc[:,1:2] # DataFrame
y = df.iloc[:,2:]
Xn = x.values # Numpy array
Yn = y.values


# SVR
# Support Vector Regression | SVR | svr da scaling edilmesi mecbur. hassas bu konuda krdşimiz.
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_scale = sc1.fit_transform(Xn)
sc2 = StandardScaler()
y_scale = np.ravel(sc2.fit_transform(Yn.reshape(-1,1)))

from sklearn.svm import SVR
svr = SVR(kernel="rbf") # Radial Based Function | rbf
svr.fit(x_scale, y_scale) # iki değer arasında bağlantı kurulması

svrPredict = svr.predict(x_scale)
plt.scatter(x_scale, y_scale, color="red")
plt.plot(x_scale, svrPredict) # Her bir x değeri için tahminde bulun
plt.show()


# SVR ile tahmin etme | ölçeklenmeden dolayı düşük sayılar çıkıyor
print(svr.predict([[11]]))
print(svr.predict([[6.6]]))



# R2
# Scaling yaptığımız için boyutu indirmemiz gerek
print("SVR - R2".center(50, "-"))
print(r2_score(y_scale, svr.predict(x_scale)))