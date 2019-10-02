import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

df = pd.read_csv("./Shad_Python_10_2/Credit.csv", sep=";", encoding="cp1251")
print(df.tail())
y = df[u'кредит']
X = df.drop(u'кредит', axis=1)

model = RandomForestClassifier(random_state=42,
                               n_estimators=30,
                               criterion='gini',
                               max_depth=5,
                               oob_score=True,
                               warm_start=False,
                               class_weight=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix = pd.DataFrame(confusion_matrix, index=model.classes_, columns=model.classes_)
print(confusion_matrix)
print('Out-of-bag score: {0}'.format(model.oob_score_))
print(metrics.classification_report(y_pred, y_test))


