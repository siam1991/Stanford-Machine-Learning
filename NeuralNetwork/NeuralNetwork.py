import scipy.io
import numpy as np

matrix = scipy.io.loadmat('ex3data1.mat')
x = matrix['X']
y = matrix['y']


thetas = scipy.io.loadmat('ex3weights.mat')
theta1 = thetas['Theta1']
theta2 = thetas['Theta2']

def sigmoid(z):
    return 1/(1+np.exp(-z))

# layer2
sample_size = x.shape[0]

x = np.hstack(([[1]]*sample_size, x))
a2 = sigmoid(np.dot(theta1, x.T))
# add a2(0)
a2 = np.hstack(([[1]]*sample_size, a2.T))

a3 = sigmoid(np.dot(theta2, a2.T))

index = np.argmax(a3, axis=0)
p = index+1
p = p.reshape(-1,1)
print('准确率：', np.sum(p == y)/sample_size * 100)