import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')

df = pd.read_csv("./Shad_Python_06_2/Albuquerque/Albuquerque Home Prices_data.txt", sep='\t')
df.replace(-9999, np.nan, inplace=True)
print(df.head())
print("Rows in the data frame: {0}".format(len(df)))
print("Rows without Nan: {0}".format(len(df.dropna(how='any'))))
print(df.apply(lambda x: sum(x.isnull()), axis=0))
del df["AGE"]
print(df.head())
df["TAX"].hist()
plt.draw()

df["TAX"] = df["TAX"].fillna(df["TAX"].mean())

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

model = LinearRegression()
model.fit(X, y)
print("R^2 : {0}".format(model.score(X, y)))

coef = pd.DataFrame(zip(['intercept'] + X.columns.tolist(), [model.intercept_] + model.coef_.tolist()), columns=['predictor', 'coef'])
print(coef)

def regression_coef(model, X, y):
    coef = pd.DataFrame(zip(['intercept'] + X.columns.tolist(), [model.intercept_] + model.coef_.tolist()), columns=['predictor', 'coef'])
    X1 = np.append(np.ones((len(X), 1)), X, axis=1)
    b = np.append(model.intercept_, model.coef_)
    MSE = np.sum((model.predict(X) - y)**2, axis=0) / float(X.shape[0])
    var_b = MSE * (np.linalg.inv(X1.T@X1).diagonal())
    sd_b = np.sqrt(var_b)
    t = b / sd_b
    coef["p_value"] = [2*(1-stats.t.cdf(np.abs(i), len(X1) - 1)) for i in t]
    return coef
print(regression_coef(model, X, y)) # we can trhow away TAX and calculate again p_values colinear

df = pd.read_csv("./Shad_AD on Python_1_02/diamond.dat", header=None, sep='\s+', names=["weight", "price"])
print(df.head())
poly = PolynomialFeatures(degree=2, include_bias=False)
y = df["price"]
X0 = poly.fit_transform(df[['weight']])
X0 = pd.DataFrame(X0, columns=['weight', 'weight^2'])
print(X0.head())
X0 = [X0[['weight']], X0[['weight^2']], X0.copy()]
models = [LinearRegression() for _ in X0]

for X, model in zip(X0, models):
    model.fit(X, y)
    print(model.score(X, y))
    print(regression_coef(model, X, y))  # we can trhow away TAX and calculate again p_values colinear

plt.tight_layout()
plt.show()
