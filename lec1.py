import os
import pandas as pd
import numpy as np

AH = pd.read_csv("./Shad_Python_01_2/Ames_dataset/AmesHousing.txt", sep='\t', header = 0, index_col=False)

print(AH.head())