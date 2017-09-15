import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))


def J_Reg(x, y, theta,lam):
    sample_size = x.shape[0]
    exp_part = np.dot(x, theta)  # m*1
    part1 = np.log(1.0 + np.exp(exp_part))
    part2 = np.dot(y.T, exp_part)
    Regualize = np.vstack((np.array([0.0]), theta[1:])) ** 2
    return 1.0 / sample_size * (np.sum(part1) - part2) + lam / 2 / sample_size * np.sum(Regualize)


def GridentDescent_Reg(x, y, theta, alpha, lam, maxstep):
    iterate = []
    error = []
    sample_size = x.shape[0]
    # ScaleFeature()
    count = 0
    while count < maxstep:
        count += 1
        J_cost = J_Reg(x, y, theta, lam)
        h = sigmoid(np.dot(x, theta))
        reg = 1 / sample_size * np.dot(x.T, (h - y)) + lam / sample_size * np.vstack((np.array([0.0]), theta[1:]))
        theta = theta - alpha * reg
        iterate.append(count)
        error.append(J_cost[0])
    # plt.plot(iterate, error)
    # plt.show()
    return theta, J_cost[0]


matrix = scipy.io.loadmat('../NeuralNetwork/ex3data1.mat')
x = matrix['X']
z = matrix['y']
num_labels = 10
sample_size = x.shape[0]
feature_size = x.shape[1]

all_theta = np.zeros((num_labels, feature_size+1))
x = np.hstack(([[1]] * sample_size, x))

for i in range(num_labels):
    y = np.where(z == i+1, 1, 0)
    theta_init = np.zeros((feature_size+1, 1))
    theta, cost = GridentDescent_Reg(x, y, theta_init, 5, 0.001, 1000)
    all_theta[i, :] = theta.T
    print(cost)

predict = sigmoid(np.dot(x, all_theta.T))
p = np.argmax(predict, axis=1)+1
p = p.reshape(-1, 1)
print(np.mean((p == z))*100)
