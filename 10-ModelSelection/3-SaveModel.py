# Pickle
# Joblib
# Pmml
import pandas as pd
from scipy.sparse.construct import random
from sklearn import linear_model

data = pd.read_csv("data/prediction1.csv")

x = data.iloc[:,0:1].values
y = data.iloc[:,1]

division = 0.33

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=division, random_state=0)


# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# print(lr.predict(x_test))

# Öğrenilen şeyleri dosyaya kaydetme
import pickle

file = "9-ModelSelection/saveModel.txt"
# pickle.dump(lr, open(file,"wb"))

uploaded = pickle.load(open(file,"rb")) 
print(uploaded.predict(x_test))

"""
İlk önce regresyonu aç ve dosyaya kaydet
Öğrendikten sonra regresyon kapat ve dosyadaki kayıtlarla devam et
"""