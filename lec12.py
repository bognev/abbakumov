import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
df = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?")
print(df.tail())
print(df.describe())
df = df.drop('education', axis=1)
df["income"] = df["income"].map({" <=50K" : 0, " >50K" : 1})
df.dropna(inplace=True)

test = pd.read_csv("adult.test", header=None, names=columns, na_values=" ?")
test = test.drop("education", axis=1)
test["income"] = test["income"].map({" <=50K." : 0, " >50K." : 1})
test.dropna(inplace=True)

print(df["income"].value_counts(normalize=True))
X_train = pd.get_dummies(df).drop('income', axis=1)
y_train = df["income"]
X_test = pd.get_dummies(test).drop("income", axis=1)
y_test = test["income"]
print(len(X_train.columns))
print(len(X_test.columns))
print(set(X_train.columns) - set(X_test.columns))
print(set(X_test.columns) - set(X_train.columns))
columns = set(X_train.columns) | set(X_test.columns)
X_train = X_train.reindex(columns=columns).fillna(0)
X_test = X_test.reindex(columns=columns).fillna(0)
print(all(X_train.columns == X_test.columns))

model = GradientBoostingClassifier(random_state=42,
                                   n_estimators=100,
                                   max_depth=3,
                                   learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))






plt.show()




