import numpy
import warnings
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score

crypto_dataset = pandas.read_csv('cryptocoin_data.csv')

print(crypto_dataset.shape)
print(crypto_dataset.info())
print(crypto_dataset.describe())

# Preprocessing - dropping categorical values, filling missing values with mean
crypto_dataset.drop(['Currency'], axis=1, inplace = True)
crypto_dataset.drop(['Date'], axis=1, inplace = True)

print(crypto_dataset)
print(crypto_dataset.info())

print('\n', crypto_dataset.isnull().sum(axis=0))
crypto_dataset.fillna(value=crypto_dataset.mean(), inplace=True)
print('\n', crypto_dataset.isnull().sum(axis=0))



scaler = MinMaxScaler()
scaledcryptoset = ['Volume', 'Market Cap']
crypto_dataset[scaledcryptoset] = scaler.fit_transform(crypto_dataset[scaledcryptoset])



sns.heatmap(crypto_dataset.corr(), annot=True)
plt.show()

# creating X and Y
X = crypto_dataset['Market Cap']
Y = crypto_dataset['Close']

# create Train and Test set


X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7,
                                                    test_size = 0.3, random_state = 100)
# adding a constant to get an intercept
X_train_sm = sm.add_constant(X_train)
# fitting regression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

print(lr.params)
print(lr.summary())

plt.scatter(X_train, y_train)
plt.plot(X_train, -137.4288 + 17330*X_train, 'r')
plt.show()

#residual analysis

y_train_pred = lr.predict(X_train_sm)

res = (y_train - y_train_pred)

# plot residual analysis using histogram
fig = plt.figure()
sns.displot(res, bins=15)
plt.title('Error Terms', fontsize = 15)
plt.xlabel('y_train - y_train_pred', fontsize = 15)
plt.show()

# Looking for any patterns in the residuals
# plt.scatter(X_train,res)
# plt.show()

# Adding a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predicting the y values corresponding to X_test_sm
y_test_pred = lr.predict(X_test_sm)


# Checking the R - squared value
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))

print("ACTUAL VALUE:",  y_test)
print("PREDICTED VALUES: ", y_train_pred)

# Visualize the line on the test set
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()


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