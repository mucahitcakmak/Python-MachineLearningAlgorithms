# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import count_empty_vals

print("PROGRAM IS STARTING".center(50, "-"))
# DATA UPLOAD 
data = pd.read_csv("data/Data.csv")


# DATA PREPROCESSİNG
# print(data)

# DATA IMPORT
country = data.iloc[:, 0:1].values
age = data.iloc[:, 1:4].values
gender = data.iloc[:, -1].values


# MISSING DATA
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(age[:, 1:4])
age[:, 1:4] = imputer.transform(age[:, 1:4])




# DATA LABEL ENCODING
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

print(country)

# DATA MERGE
categoryResult = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "us"])
# print(categoryResult)

missingDataResult = pd.DataFrame(data=age, index=range(22), columns=["lenght", "weight", "age"])
# print(missingDataResult)

genderResult = pd.DataFrame(data=gender, index=range(22), columns=["gender"])
# print(genderResult)


# tek tek değiştirip özelleştirdiğimiz listeleri
# birleştiriyoruz
s = pd.concat([categoryResult, missingDataResult, genderResult], axis=1)
print(s)



print("PROGRAM IS FINISHED".center(50, "-"))