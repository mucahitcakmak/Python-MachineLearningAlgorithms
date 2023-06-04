import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import random

data = pd.read_csv("data/Ads_CTR_Optimisation.csv")


# Thompson
N = 10000 # 10000 tıklama, reklam gösterimi
d = 10 # 10 ilan var

total = 0 # Toplam ödül
choisen = []
ones = [0] * d
zeros = [0] * d
for n in range(1,N):
    ad = 0 # Seçilen ilan
    max_th = 0
    for i in range(0,d):
        ranBeta = random.betavariate(ones[i] + 1, zeros[i] + 1)  
        if ranBeta > max_th:
            max_th = ranBeta
            ad = i
    choisen.append(ad)
    reward = data.values[n,ad]
    if reward == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] + 1 
    total = total + reward


print("Total Reward:",total)

plt.hist(choisen)
# plt.show()