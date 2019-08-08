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
plt.hist(AH['SalePrice'], bins=60, log=True)#.log.hist(bins=60, density=1)

from scipy.stats.kde import gaussian_kde
from numpy import linspace, hstack
from pylab import plot, show, hist

plt.show()
# print(AH.head())