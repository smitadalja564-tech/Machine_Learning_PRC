import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

salary = pd.read_csv(r"D:\Data Science\Machine Leaning\CSV\Salary_dataset.csv")

salary.describe()

a1 = salary.drop(columns = ['Unnamed: 0'])

# print(a1)

# print(a1.describe())
# print(a1.info())
# print(a1.isnull().sum())

# Scatter plot

plot = a1.plot.scatter(x = 'YearsExperience', y = 'Salary') 
# plt.show()


X = a1[['YearsExperience']]
Y = a1['Salary']

# check it shape

# print(X.shape)
# print(Y.shape)

# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 47)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)

print(model.coef_)
print(model.intercept_)

ypred = model.predict(X_test)
print(Y_test)
print(ypred)

New = X_test
New['Salary'] = Y_test
New['Predicted Salary'] = ypred

print(New)

New.to_csv(r"D:\\Data Science\\Machine Leaning\\CSV\\Salary_Predicted.csv", index = False)

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, root_mean_squared_error

print(mean_squared_error(Y_test, ypred))

print(mean_absolute_error(Y_test, ypred))

print(mean_absolute_percentage_error(Y_test, ypred))

print(root_mean_squared_error(Y_test, ypred))

from sklearn.metrics import r2_score

print(r2_score(Y_test, ypred))


q = np.array([[2.1],[3.5]])
print(q.shape)