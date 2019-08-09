import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

AH = pd.read_csv("./Shad_Python_01_2/Ames_dataset/AmesHousing.txt", sep='\t', header = 0, index_col=False)

print(AH.shape)
print(len(AH))
print(AH.dtypes)
print(AH.describe(include='all'))
# AH['SalePrice'].hist()
print(style.available)
# fig, ax = plt.subplots()
# plt.hist(AH['SalePrice'], bins=60, log=True)#.log.hist(bins=60, density=1)

from scipy.stats.kde import gaussian_kde
from numpy import linspace, hstack
from pylab import plot, show, hist

my_density = gaussian_kde(AH['SalePrice'])
x = linspace(min(AH['SalePrice']), max(AH['SalePrice']), 1000)
# plot(x, my_density(x), 'g')
# hist(AH['SalePrice'], normed=1, alpha=0.3)
# AH.groupby('MS Zoning')['SalePrice'].plot.hist(density=1, alpha=0.6)
ax = AH.boxplot(column='SalePrice', by='MS Zoning')
ax.get_figure().suptitle('')
plt.show()
# print(AH.head())