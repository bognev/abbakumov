import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.plotting import scatter_matrix
style.use('ggplot')
df = pd.read_csv('./Shad_Python_01_2/Swiss/Swiss Bank Notes.dat', sep=' ', header=0, index_col=False)
# print(df['Length'])
df['Status'] = 100*['genuine'] + 100*['counterfeit']

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe(include='all'))

# df.groupby('Status')['Length'].plot.hist(alpha=0.6)

df.groupby('Status')['Diagonal'].plot.hist(alpha=0.6)
plt.legend(loc='upper left')




colors = {'genuine' : 'green', 'counterfeit' : 'red'}
df.plot.scatter(x='Top', y='Bottom', c=df['Status'].replace(colors))
scatter_matrix(df,
               figsize=(6,6),
               diagonal='kde',
               c=df['Status'].replace(colors),
               alpha=0.2)


plt.show()