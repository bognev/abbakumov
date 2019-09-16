import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import style
style.use("ggplot")

df = pd.read_csv("./home/zabolotsky/PycharmProjects/abbakumov/Shad_AD on Python_1_01/2_pretendent/assess.dat", sep="\t", index_col='NAME')
print(df.head())
