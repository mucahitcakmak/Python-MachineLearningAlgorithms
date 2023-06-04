import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
# Rastgele seçim yapar
# Öğrenme veya bi zeka yoktur 
# UCB nin random versiyonu gibi ama öğrenmiyp


data = pd.read_csv("data/Ads_CTR_Optimisation.csv")


# Random Selection
import random
N = 10000 # Kaça kadar rastgele sayı üretsin
d = 10 # kaçla kaç arasında üretsin 
total = 0
choisen = []
for n in range(0, N):
    ad = random.randrange(d) 
    choisen.append(ad)
    reward = data.values[n, ad] # Verilerdeki n. satırı = 1 ise ödül 1 | 0 ise ödül 0
    total = total + reward  

print(total)
plt.hist(choisen)
# plt.show()
