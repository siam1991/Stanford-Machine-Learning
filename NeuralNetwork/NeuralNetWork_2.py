import numpy as np
import scipy.io
import pandas as pd


def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def CostFunction(x,y,thetas,num_labels,lamb):
    theta1 = thetas['Theta1']
    theta2 = thetas['Theta2']
    sample_size = x.shape[0]
    # warning:numpy中以下代码会修改原始theta
    # theta_1 = theta1
    # theta_1[:,0] = 0
    # theta_2 = theta2
    # theta_2[:,0] = 0
    # theta1_sqrt = theta_1**2
    # theta2_sqrt = theta_2**2
    theta_1 = np.hstack(([[0]]*theta1.shape[0], theta1[:, 1:]))
    theta_2 = np.hstack(([[0]]*theta2.shape[0], theta2[:, 1:]))

    x = np.hstack(([[1]]*sample_size, x))
    z2 = np.dot(theta1, x.T)
    a2 = sigmoid(z2)
    a2 = np.hstack(([[1]]*sample_size, a2.T))
    z3 = np.dot(theta2, a2.T)
    h = sigmoid(z3)
    y_vector = np.zeros((num_labels, sample_size))
    for i in range(sample_size):
        y_vector[y[i]-1,i]=1
    part1 = y_vector*np.log(h)
    part2 = (1-y_vector)*np.log(1-h)
    J = -1/sample_size*(np.sum(part1)+np.sum(part2))+lamb/2/sample_size*(np.sum(theta_1**2)+np.sum(theta_2**2))


    theta1_grad = np.zeros_like(theta1)
    theta2_grad = np.zeros_like(theta2)
    delta_3 = h-y_vector
    z2 = np.hstack(([[1]]*sample_size, z2.T))
    delta_2 = np.dot(theta2.T, delta_3)*sigmoidGradient(z2.T)
    delta_2 = delta_2[1:,:]
    theta1_grad = 1/sample_size*(theta1_grad+np.dot(delta_2, x))+lamb/sample_size*theta_1
    theta2_grad = 1/sample_size*(theta2_grad+np.dot(delta_3, a2)+lamb/sample_size*theta_2)
    return J, theta1_grad, theta2_grad




if __name__ == '__main__':
    matrix = scipy.io.loadmat('ex4data1.mat')
    thetas = scipy.io.loadmat('ex4weights.mat')
    x = matrix['X']
    y = matrix['y']
    cost, theta1_grad, theta2_grad = CostFunction(x, y, thetas, 10, 1)
    print(cost)
    print(theta1_grad)
    print(theta2_grad)




