import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
print(df.head())
print(df.describe())

fix, ax = plt.subplots(0)
df['Survived'].value_counts().plot(kind='bar')
fig, ax2 = plt.subplots(1)
df['Survived'].value_counts().plot(kind='pie')
plt.show()