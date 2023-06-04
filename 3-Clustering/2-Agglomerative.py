import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/customers.csv")

x = data.iloc[:,3:].values


# HC Agglomerative
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
y_pred = ac.fit_predict(x)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c="red")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c="green")
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c="blue")
plt.scatter(x[y_pred==3,0], x[y_pred==3,1], s=100, c="yellow")
plt.show()