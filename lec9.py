import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt
style.use("ggplot")
df = pd.read_csv("./Shad_AD on Python_1_03/series_g.csv", sep=';')
print(df.head())
print(df.tail())
df["date"] = pd.to_datetime(df['date'], format='%b %Y')
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
df['series_g'].plot(ax=ax1)
ax1.set_title(u'Объем пассажироперевозок')
ax1.set_ylabel(u'Тысяч человек')
ax2 = fig.add_subplot(122)
pd.Series(np.log10(df['series_g'])).plot(ax=ax2)
ax2.set_title(u'log10 Объем пассажироперевозок')
ax2.set_ylabel(u'log10 Тысяч человек')

new_dates = pd.date_range('1961-01-01', '1961-12-01', freq='MS')
new_dates = pd.Index(df['date']) | new_dates
df2 = pd.DataFrame({'date':new_dates})
df = pd.merge(df, df2, on='date', how='right')
df['month_num'] = range(1,len(df) + 1)
df['log_y'] = np.log10(df['series_g'])

for x in range(1, 13):
    df['season_' + str(x)] = df['date'].dt.month == x

season_columns = ['season_' + str(x) for x in range(2,13)]

X = df[['month_num'] + season_columns]
y = df['log_y']

X1 = X[X.index < 144]
y1 = y[y.index < 144]

model = LinearRegression()
model.fit(X1,y1)

pred = pd.DataFrame({'pred' : model.predict(X1), 'real' : y1})
pred.plot()

pred = pd.DataFrame({'pred' : model.predict(X), 'real' : y})
pred.plot()

plt.show()
