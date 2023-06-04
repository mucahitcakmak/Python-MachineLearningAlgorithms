import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import math

data = pd.read_csv("data/Ads_CTR_Optimisation.csv")


# UCB (Upper Confidence Bound) | (Üst Güven Sınırı)
N = 10000 # 10000 tıklama, reklam gösterimi
d = 10 # 10 ilan var
rewards = [0] * d # İlk başta bütün ilanların ödülü sıfır
clicks = [0] * d # Tüm Tıklamalar
total = 0 # Toplam ödül
choisen = []
for n in range(1,N):
    ad = 0 # Seçilen ilan
    max_ucb = 0

    for i in range(0,d):
        if clicks[i] > 0:
            average = rewards[i] / clicks[i]
            delta = math.sqrt(3/2 * math.log(n) / clicks[i])
            ucb = average + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i

    choisen.append(ad)
    clicks[ad] = clicks[ad] + 1 
    reward = data.values[n,ad]
    rewards[ad] = rewards[ad] + reward
    total = total + reward


print("Total Reward:",total)

plt.hist(choisen)
# plt.show()