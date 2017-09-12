from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# data1
print("ex2data1")
dataframe = pd.read_csv('ex2data1.txt', header=None)
columns = dataframe.shape[1]-1
sample_size = dataframe.shape[0]
x = dataframe[list(range(columns))]
z = dataframe[columns]
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x = np.hstack(([[1]] * x.shape[0], x))
model = linear_model.LogisticRegression()
model.fit(x, z)
print(model.coef_)
p = np.dot(x, model.coef_[0])
p = np.where(p >= 0.5, 1, 0)
correct_percent = np.sum(p == z) / sample_size * 100  # np.mean(p==self.z)*100
print("Train Accuracy:", correct_percent)
x_test = scaler.transform(np.array([[55.48216114069585, 35.57070347228866]]))
x_test = np.hstack(([[1]] * x_test.shape[0], x_test))
result = model.predict(x_test)
print(result)

# data2
print("ex2data2")
dataframe = pd.read_csv('ex2data2.txt', header=None)
sample_size = dataframe.shape[0]
columns = dataframe.shape[1]-1
x = dataframe[list(range(columns))]
z = dataframe[columns]
poly = PolynomialFeatures(6)
x = poly.fit_transform(x)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
model = linear_model.LogisticRegression()
model.fit(x, z)
print(model.coef_)
p = np.dot(x, model.coef_[0])
p = np.where(p >= 0.5, 1, 0)
correct_percent = np.sum(p == z) / sample_size * 100  # np.mean(p==self.z)*100
print("Train Accuracy:", correct_percent)
x_test = poly.fit_transform(np.array([[0.13191, -0.51389]]))
x_test = scaler.transform(x_test)
result = model.predict(x_test)
print(result)


