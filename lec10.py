import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
model = DecisionTreeClassifier(random_state=42,
                               criterion='gini',
                               max_depth=5,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               class_weight=None,
                               presort=False)
model.fit(X,y)
export_graphviz(model,
                out_file="tree.dot",
                class_names=None,
                label="all",
                filled=True,
                impurity=True,
                node_ids=True,
                proportion=True,
                rotate=False)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])
img=mpimg.imread('tree.png')
imgplot = plt.imshow(img)

print(pd.DataFrame({'feature' : X.columns, 'importance': model.feature_importances_}).sort_values('importance'))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix = pd.DataFrame(confusion_matrix, index=model.classes_, columns=model.classes_)
print(confusion_matrix)
tp = confusion_matrix[0][0]
fp = confusion_matrix[0][1]
fn = confusion_matrix[1][0]
tn = confusion_matrix[1][1]
#tp fp
#fn tn
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print("precision = {}".format(precision))
print("recall = {}".format(recall))
f1 = 2*precision*recall/(precision + recall)
print("f1-score = {}".format(f1))
print(metrics.classification_report(y_pred, y_test))



plt.show()

