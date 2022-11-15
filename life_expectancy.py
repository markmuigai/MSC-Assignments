import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('life_exp.csv')

# Find Correlation
# plt.figure(figsize= (14,10))
# sns.heatmap(dataset.corr(), annot=True)
# plt.show()

print('\n Data shape:  ', dataset.shape)
print('\n Data Columns:  ', dataset.shape)
print('\n ', dataset.info())
print('\n', dataset.isnull().sum(axis=0))

dataset.fillna(value=dataset.mean(), inplace=True)
print('\n', dataset.isnull().sum(axis=0))

X = dataset[['Total expenditure','GDP']].values
Y = dataset['Life expectancy'].values

print(X)
print(X.shape)
print(Y)
print(Y.shape)

validation_size = 0.10
seed = 9

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)

#  let the model predict X_validation
predictions = model_LR.predict(X_validation)
print('R squared ', r2_score(Y_validation, predictions))
print("Root Mean Squared", numpy.sqrt(mean_squared_error(Y_validation, predictions)))
print("Mean Absolute Error", numpy.sqrt(mean_absolute_error(Y_validation, predictions)))

print(Y_validation)
print(predictions)