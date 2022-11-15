import pandas
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

crypot_dataset = pandas.read_csv('cryptocoin_data.csv')


print(crypot_dataset.info())
print(crypot_dataset.corr())

x = crypot_dataset['High']
y = crypot_dataset['Close']


def linear_regression(x, y):
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean) ** 2).sum()
    B1 = B1_num / B1_den

    B0 = y_mean - (B1 * x_mean)

    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))

    return (B0, B1, reg_line)


def corr_coef(x, y):
    N = len(x)

    num = (N * (x * y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x * 2).sum() - x.sum() * 2) * (N * (y * 2).sum() - y.sum() * 2))
    R = num / den
    return R


B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)


def predict(B0, B1, new_x):
    y = B0 + B1 * new_x
    return y


new_x = 0.3112
predict_output = predict(B0, B1, new_x)
print(new_x)
# plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
# sns.lmplot(x='High', y='Close', data=crypot_dataset)
# plt.title("Scatter Plot with Linear fit");
# plt.show()