import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import  variance_inflation_factor

from statsmodels.stats.outliers_influence import variance_inflation_factor

dataset = pandas.read_csv('life_exp.csv')

# description of dataset
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# Encoding of categorical values
Status = pandas.get_dummies(dataset['Status'])
print(Status)

dataset = pandas.concat([dataset, Status], axis = 1)
# print(dataset)

# replacing missing value with mean and dropping Categorical values

# print('\n', dataset.isnull().sum(axis=0))
dataset.drop(['Country'], axis=1, inplace = True)
dataset.drop(['Status'], axis=1, inplace = True)
dataset.drop(['Year'], axis=1, inplace = True)
dataset.fillna(value=dataset.mean(), inplace=True)
# print('\n', dataset.isnull().sum(axis=0))

# print(dataset.shape)

# split data into 7:3 ratio

dataset_train, dataset_test = train_test_split(dataset, train_size = 0.7, test_size = 0.3, random_state = 100)

# Re-scaling features
scaler = MinMaxScaler()
num_dataset = ['Life expectancy', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
               'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP',
               'Population', 'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources', 'Schooling']
dataset_train[num_dataset] = scaler.fit_transform(dataset_train[num_dataset])


# divide data set into X and Y sets

Y_train = dataset_train.pop('Life expectancy')
X_train = dataset_train

# fit the model with the train set to test for p-value
X_train_lm = sm.add_constant(X_train)
lr_model = sm.OLS(Y_train, X_train_lm).fit()

# print(lr_model.summary())

# ----testing to remove muliti collinearlity
vif = pandas.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
# print(vif)

X = X_train.drop(['thinness 5-9 years', 'Population','percentage expenditure', 'Developing', 'Developed',
                  ], axis=1 )
# Build a fitted model after dropping the variable
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(Y_train, X_train_lm).fit()

# Printing the summary of the model
# print(lr_2.summary())

vif = pandas.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
# print(vif)

X = X.drop(['under-five deaths', 'infant deaths','Schooling', 'Income composition of resources', 'Diphtheria',
                 'Polio', 'Hepatitis B', 'BMI','Total expenditure', ], axis=1 )
# Build a fitted model after dropping the variable
X_train_lm = sm.add_constant(X)

lr_3 = sm.OLS(Y_train, X_train_lm).fit()

# Printing the summary of the model
# print(lr_3.summary())

vif = pandas.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
# print(vif)




# Residual analysis
Y_train_expectancy = lr_3.predict(X_train_lm)
fig = plt.figure()
sns.displot((Y_train - Y_train_expectancy), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading
plt.xlabel('Errors', fontsize = 1)
# plt.show()


def linear_regression(x, y):
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

# calculating the  intercept
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean) ** 2).sum()
    B1 = B1_num / B1_den

# calculating the constant incercept
    B0 = y_mean - (B1 * x_mean)

    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))

    return (B0, B1, reg_line)


def corr_coef(x, y):
    N = len(x)

    num = (N * (x * y).sum()) - (x.sum() * y.sum())
    den = numpy.sqrt((N * (x * 2).sum() - x.sum() * 2) * (N * (y * 2).sum() - y.sum() * 2))
    R = num / den
    return R


# B0, B1, reg_line = linear_regression(X, Y_train)
# print('Regression Line: ', reg_line)
# R = corr_coef(X,  Y_train)
# print('Correlation Coef.: ', R)
# print('Goodness of Fit": ', R**2)

# Making predictions using the selected variables
dataset_test[num_dataset] = scaler.transform(dataset_test[num_dataset])

# Divide test data into X and Y
Y_test = dataset_test.pop('Life expectancy')
X_test = dataset_test

X_test_m4 = sm.add_constant(X_test)
X_test_m4 = X_test_m4.drop(['under-five deaths', 'infant deaths','Schooling', 'Income composition of resources',
                            'Diphtheria', 'Polio', 'Hepatitis B', 'BMI','Total expenditure', 'thinness 5-9 years',
                            'Population','percentage expenditure', 'Developing', 'Developed'], axis = 1)


Y_predict_m4 = lr_3.predict(X_test_m4)

r2_score = r2_score(y_true=Y_test, y_pred=Y_predict_m4)
print("R2 score: ", r2_score)


print("Root Mean Squared", numpy.sqrt(mean_squared_error(Y_test, Y_predict_m4)))
print("Mean Absolute Error", numpy.sqrt(mean_absolute_error(Y_test, Y_predict_m4)))
print("MSE: ", mean_squared_error(Y_test, Y_predict_m4))

print("ACTUAL VALUE:",  Y_test)
print("PREDICTED VALUES: ", Y_predict_m4)