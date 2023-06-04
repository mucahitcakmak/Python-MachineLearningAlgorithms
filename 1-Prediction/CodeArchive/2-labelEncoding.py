# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("PROGRAM IS STARTING".center(50, "-"))

# DATA UPLOAD 
data = pd.read_csv("Data.csv")


# DATA LABEL ENCODING
country = data.iloc[:, 0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)









print("PROGRAM IS FINISHED".center(50, "-"))
