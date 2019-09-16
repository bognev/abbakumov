import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("ggplot")
import os

df = pd.read_csv("./Shad_AD on Python_1_01/1_beverage/beverage_r.csv", sep=";", index_col='numb.obs')

print(df.tail())
print(df.shape)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
link = linkage(df, 'ward', 'euclidean') #кластерный анализ #расстояние между кластерами и #объектами
print(link)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(link, orientation="right")

df['cluster'] = fcluster(link, 3, criterion='distance')
print(df.groupby('cluster').mean())#какая доля людей пила данный напиток в данном кластере
print(df.groupby('cluster').size())
plt.show()


