import pandas as pd
import numpy as np
import seaborn as sns
# from sklearn.model_selection import trai

dataset = pd.read_csv('life_exp.csv');
dataset.drop('Country', inplace=True, axis=1)
dataset.drop('Status', inplace=True, axis=1)

var =np.corrcoef(dataset)
print(var)

