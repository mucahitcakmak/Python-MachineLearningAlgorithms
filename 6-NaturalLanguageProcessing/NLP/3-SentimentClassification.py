import pandas as pd 
import numpy as np



# Import Dataset
data = pd.read_csv("data/newsSentencesA1.csv")

text = data[["text"]] # DataFrame
value = data[["label"]].values # Numpy array


# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
value[:, 0] = le.fit_transform(value[:, 0])


# Data Merge
valueResult = pd.DataFrame(data=value, columns=["label"])
newData = pd.concat([text, valueResult], axis=1)


# Word Cleaning
import string
import re
import nltk
from nltk.corpus import stopwords

punctation = string.punctuation
stopWords = stopwords.words("turkish")




for text in newData["text"].head():
    print(text + "\n-------------------")
    temp = ""
    for word in text.split():
        if word not in stopWords and not word.isnumeric():
            temp += word + " "

    temp = " "
    for word in text:   
        if word not in punctation:
            temp += word
    print(temp + "\n*************************************************")








