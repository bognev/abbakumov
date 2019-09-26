import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from scipy import stats

#                     hypothesis true hypothesis false
#hypothesis accepted        +        2nd type error
#hypothesis rejected  1st type error        +
#alpha significance = 0.05, 0.01, 0.005 margin from top for frequency of 1st type mistakes
#it will give us 1 mistake from 20, 100, 200 checks

df = pd.read_csv("./Shad_Python_06_2/town_1959_2.csv", encoding='cp1251')
print(df.tail())
df = df.set_index(u'номер')

res = stats.shapiro(np.log(df[u'население']))
print('p-value: ', res[1])

fig, ax = plt.subplots(2,2, figsize=(10,5))
ax[0,0].hist(df[u'население'], bins=50)
ax[0,1].hist(np.log10(df[u'население']), bins=50)
plt.tight_layout()
plt.draw()

df = pd.read_csv("./Shad_Python_06_2/Albuquerque/Albuquerque Home Prices_data.txt", sep='\t')
print(df.head())
df.replace(-9999, np.nan, inplace=True)
print(df.head())
x=df[df['COR']==1]['PRICE']
y=df[df['COR']==0]['PRICE']
x.name, y.name = 'corner', 'not corner'

def two_histogramms



#plt.show()

