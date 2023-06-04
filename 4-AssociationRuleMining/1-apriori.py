import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("data/hamper.csv", header=None)


# Her veriyi liste içine sokuyoruz (sutün isimleri yok olması gerekli değil)
t = []
for i in range(0, 7501): # satır sayısı
    t.append([str(data.values[i,j]) for j in range(0, 20)]) # en fazla sütun olan sayı



# Apriori
from apyori import apriori
rules = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_lenght=2)


# print(list(rules))