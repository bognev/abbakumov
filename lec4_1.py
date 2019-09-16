import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")

df = pd.read_csv("/home/zabolotsky/PycharmProjects/abbakumov/Shad_AD on Python_1_01/1_beverage/beverage_r.csv", sep=';', header = 0, index_col=False)
print(df)
print(df.describe())
model = KMeans(n_clusters=3, verbose=0)
model.fit(df)
print(model.labels_)
print(model.cluster_centers_)
new_items = [[1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0]]
print(model.predict(new_items))
K = range(1,11)
models = [KMeans(n_clusters=k, random_state=42, verbose=0).fit(df) for k in K]
dist = [model.inertia_ for model in models]

df['cluster'] = model.labels_
mns = df.groupby('cluster').mean()
print(mns)
print(df.groupby('cluster').size())

fig = plt.figure()
plt.plot(K, dist, marker='o')
plt.show()



