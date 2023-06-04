# LIBRARYS ***************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("PROGRAM IS STARTING".center(50, "-"))
# DATA UPLOAD 
data = pd.read_csv("data/firstData.csv")


# DATA PREPROCESSİNG
# print(data)

# DATA IMPORT
country = data.iloc[:, 0:1].values
age = data.iloc[:, 1:4].values
gender = data.iloc[:, -1].values


# MISSING DATA
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(age)
age = imputer.transform(age)




# DATA LABEL ENCODING
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:, 0] = le.fit_transform(data.iloc[:, 0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()



# DATA MERGE
categoryResult = pd.DataFrame(data=country, index=range(22), columns=["fr", "tr", "us"])
# print(categoryResult)
missingDataResult = pd.DataFrame(data=age, index=range(22), columns=["lenght", "weight", "age"])
# print(missingDataResult)
genderResult = pd.DataFrame(data=gender, index=range(22), columns=["gender"])
# print(genderResult)

# tek tek değiştirip özelleştirdiğimiz listeleri
# birleştiriyoruz
s = pd.concat([categoryResult, missingDataResult], axis=1)
s2 = pd.concat(genderResult, axis=1)
# print(s)

# Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,s2, test_size=0.33, random_state=0)

# Verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
print




print("PROGRAM IS FINISHED".center(50, "-"))