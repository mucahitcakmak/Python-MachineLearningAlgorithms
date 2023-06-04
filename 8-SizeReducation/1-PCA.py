import pandas as pd
import numpy as np
# Veriyi sıkıştırmak için pca yapıyoruz daha az yer kaplasın diye

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


# PCA | Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 2 Boyuta indirgedi | 13 kolon vardı 2 tane oldu | magic
X_train2 = pca.fit_transform(X_train) # ama veri kaybı olabilir 
X_test2 = pca.transform(X_test) # ama veri kaybı olabilir


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
print("CM (No PCA):")
print(cm)

cm2 = confusion_matrix(y_test, y_pred2)
print("CM (Yes PCA):")
print(cm2)

