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


def two_histograms(x, y):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(x, alpha=0.5, weights=[1./len(x)]*len(x), label='x')
    ax.hist(y, alpha=0.5, weights=[1./len(y)]*len(y), label='y')
    plt.tight_layout()
    plt.draw()

df = pd.read_csv("./Shad_Python_06_2/town_1959_2.csv", encoding='cp1251')
print(df.tail())
df = df.set_index(u'номер')

res = stats.shapiro(np.log(df[u'население']))
print('p-value: ', res[1])

fig, ax = plt.subplots(2,1, figsize=(5,3))
ax[0].hist(df[u'население'], bins=100)
ax[1].hist(np.log10(df[u'население']), bins=50)
plt.tight_layout()
plt.draw()

df = pd.read_csv("./Shad_Python_06_2/Albuquerque/Albuquerque Home Prices_data.txt", sep='\t')
print(df.head())
df.replace(-9999, np.nan, inplace=True)
print(df.head())
x=df[df['COR']==1]['PRICE']
y=df[df['COR']==0]['PRICE']
x.name, y.name = 'corner', 'not corner'
two_histograms(x, y)
res = stats.mannwhitneyu(x, y)
print('p-value: ', res[1])

df = pd.read_csv("./Shad_Python_06_2/agedeath.dat.txt", sep='\s+', header=None, names=['group', 'age', 'index'])
print(df.head())
x = df[df['group'] == 'sovr']['age']
y = df[df['group'] == 'aris']['age']
two_histograms(x, y)
res = stats.fligner(x, y)
print('p-value: ', res[1])
res = stats.ttest_ind(x, y, equal_var=False)
print('p-value: ', res[1])

df = pd.read_csv("./Shad_Python_06_2/interference.csv")
print(df.head())
x = df['DiffCol']
y = df['Black']
x.name, y.name = 'DiffCol', 'Black'
two_histograms(x, y)
res = stats.fligner(x, y)
print('p-value: ', res[1])
res = stats.ttest_ind(x, y, equal_var=False)
print('p-value: ', res[1])

contingency_table = pd.DataFrame([[28, 72], [20, 80]], index=['city', 'country'], columns=['for', 'against'])
print(contingency_table)
res = stats.chi2_contingency(contingency_table)
print('p-value: ', res[1])

df = pd.read_csv("./Shad_Python_06_2/Albuquerque/Albuquerque Home Prices_data.txt", sep='\t')
df.replace(-9999, np.nan, inplace=True)
print(df.head())
fix, ax = plt.subplots()
ax.scatter(df['PRICE'], df['SQFT'])
plt.draw()
res = stats.pearsonr(df['PRICE'], df['SQFT'])
print('pearson rho: ', res[0])
print('p-value: ', res[1])
res = stats.pearsonr(df['TAX'], df['SQFT'])
print('pearson rho: ', res[0])
print('p-value: ', res[1])

plt.show()

