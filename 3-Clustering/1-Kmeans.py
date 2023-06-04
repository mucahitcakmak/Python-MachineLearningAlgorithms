import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/customers.csv")

x = data.iloc[:,3:].values


# K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(x)

# print(kmeans.cluster_centers_)

results = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=123)
    kmeans.fit(x)
    results.append(kmeans.inertia_) # wcss değerler


plt.plot(range(1,11), results) # Dirsek noktası hangisiyse o  en iyisidir (sayılır) | 2 veya 4 clusters iyi gibi
plt.show()
