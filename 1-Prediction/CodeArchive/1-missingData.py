# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("PROGRAM IS STARTING".center(50, "-"))
# DATA UPLOAD 
data = pd.read_csv("data/firstData.csv")


# DATA PREPROCESSİNG
# print(data)

# MISSING DATA
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

age = data.iloc[:, 1:4].values
# print(age)
imputer = imputer.fit(age)
age = imputer.transform(age)
print(age)









print("PROGRAM IS FINISHED".center(50, "-"))
