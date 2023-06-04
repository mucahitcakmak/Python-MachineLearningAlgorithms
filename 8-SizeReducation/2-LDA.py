import pandas as pd
import numpy as np
# PCA sınıf farkı yoktur(gözetimsiz) | LDA sınıf farkı vardır(gözetimli)

data = pd.read_csv("data/wine.csv")

x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# LDA | Linear Discrimiant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2) # 2 Boyuta indirge

X_train2 = lda.fit_transform(X_train, y_train) # 2 değer giriyoruz
X_test2 = lda.transform(X_test) 


# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) # Pca dönüşüm öncesi | 13 kolon
classifier.fit(X_train, y_train)

classifier2 = LogisticRegression(random_state=0)# Pca dönüşüm sonrası | 2 kolon
classifier2.fit(X_train2, y_train)

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("CM (No LDA):")
print(cm)

cm2 = confusion_matrix(y_test, y_pred2)
print("CM (Yes LDA):")
print(cm2)

