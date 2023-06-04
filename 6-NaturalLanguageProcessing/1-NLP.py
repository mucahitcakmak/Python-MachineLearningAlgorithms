from nltk.util import pr
import pandas as pd
import numpy as np
import re
import nltk 


comments = pd.read_csv("data/restaurantReviews.csv")


# PreProccessing | Kelime temizleme işlemi yapıldı
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
stopWords = nltk.download("stopwords")
derlem = []
for i in range(0,20):
    comment = re.sub("[^a-zA-Z]"," ", comments["Review"][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words("english"))]
    comment = " ".join
    derlem.append(comment)



# Feature Extraction ( Öznitelik Çıkarımı ) | Bag Of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=300)
x = cv.fit_transform(derlem).toarray() # Bağımsız
y = comments.iloc[:,1].values # Bağımlı


# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20) 


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

