from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

dataframe = pd.read_csv('ex1data2.txt', header=None)
columns = dataframe.shape[1]-1
x = dataframe[list(range(columns))]
z = dataframe[columns]


scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = np.hstack(([[1]] * x.shape[0], x))
model = linear_model.LinearRegression()
model.fit(x, z)
print(model.coef_)
x_test = scaler.transform(np.array([[1650, 3]]))
x_test = np.hstack(([[1]] * x_test.shape[0], x_test))
result = model.predict(x_test)
print(result)
