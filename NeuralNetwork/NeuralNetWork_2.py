import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt


def InitializeTheta(input_layer, hidden_layer):
    epsilon_init = 0.12
    return np.random.rand(hidden_layer, input_layer+1)*2*epsilon_init-epsilon_init

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))

# only calculate J
def J_cost(x, y,thetas, input_size, hidden_size, num_labels, lamb):
    sample_size = x.shape[0]

    theta1 = thetas[0:(input_size + 1) * hidden_size].reshape(hidden_size, input_size + 1)
    theta2 = thetas[(input_size + 1) * hidden_size:].reshape(num_labels, hidden_size + 1)

    theta_1 = np.hstack(([[0]] * theta1.shape[0], theta1[:, 1:]))
    theta_2 = np.hstack(([[0]] * theta2.shape[0], theta2[:, 1:]))

    x = np.hstack(([[1]] * sample_size, x))
    z2 = np.dot(theta1, x.T)
    a2 = sigmoid(z2)
    a2 = np.hstack(([[1]] * sample_size, a2.T))
    z3 = np.dot(theta2, a2.T)
    h = sigmoid(z3)
    y_vector = np.zeros((num_labels, sample_size))
    for i in range(sample_size):
        y_vector[y[i] - 1, i] = 1
    part1 = y_vector * np.log(h)
    part2 = (1 - y_vector) * np.log(1 - h)
    J = -1 / sample_size * (np.sum(part1) + np.sum(part2)) + lamb / 2 / sample_size * (
    np.sum(theta_1 ** 2) + np.sum(theta_2 ** 2))
    return J


def ComputeNumericalGradient(x, y, thetas, input_size, hidden_size, num_labels, lamb):
    delta = 1e-4
    thetas_gradient = np.zeros_like(thetas)
    thetas_delta = np.zeros_like(thetas)
    for i in range(thetas.shape[0]):
        thetas_delta[i] = delta
        J_add = J_cost(x, y, thetas+thetas_delta, input_size, hidden_size, num_labels, lamb)
        J_minus = J_cost(x, y, thetas-thetas_delta, input_size, hidden_size, num_labels, lamb)
        thetas_gradient[i] = (J_add-J_minus)/2/delta
        thetas_delta[i] = 0
    return thetas_gradient


def GradientCheck(thetas_nu,thetas_derive):
    print(list(zip(thetas_derive, thetas_nu)))
    print(thetas_derive-thetas_nu)


def GridentDescent(x,y,thetas, input_size, hidden_size, num_labels, alpha, lamb, max_iter):
    error_record = []
    iter_record = []
    for i in range(max_iter):
        cost, thetas_grad = J(x, y, thetas, input_size, hidden_size, num_labels, lamb)
        error_record.append(cost)
        iter_record.append(i)
        thetas = thetas-alpha*thetas_grad
    return thetas, error_record,iter_record


def J(x, y, thetas, input_size, hidden_size, num_labels, lamb):
    sample_size = x.shape[0]
    # warning:numpy中以下代码会修改原始theta
    # theta_1 = theta1
    # theta_1[:,0] = 0
    # theta_2 = theta2
    # theta_2[:,0] = 0
    # theta1_sqrt = theta_1**2
    # theta2_sqrt = theta_2**2
    # print((input_size+1)*hidden_size-1)
    theta1 = thetas[0:(input_size+1)*hidden_size].reshape(hidden_size, input_size+1)
    theta2 = thetas[(input_size+1)*hidden_size:].reshape(num_labels, hidden_size + 1)

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
        y_vector[y[i]-1, i]=1
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
    # 合并
    thetas_grad = np.hstack((theta1_grad.reshape((input_size+1)*hidden_size), theta2_grad.reshape(num_labels*(hidden_size + 1))))
    return J, thetas_grad


def Predict(x,y,thetas,input_size, hidden_size, num_labels):
    theta1 = thetas[0:(input_size + 1) * hidden_size].reshape(hidden_size, input_size + 1)
    theta2 = thetas[(input_size + 1) * hidden_size:].reshape(num_labels, hidden_size + 1)
    sample_size = x.shape[0]
    x = np.hstack(([[1]] * sample_size, x))
    z2 = np.dot(theta1, x.T)
    a2 = sigmoid(z2)
    a2 = np.hstack(([[1]] * sample_size, a2.T))
    z3 = np.dot(theta2, a2.T)
    h = sigmoid(z3)
    p = np.argmax(h, axis=0)+1
    p = p.reshape(-1, 1)
    print('准确率：', np.sum(p == y) / sample_size * 100)



if __name__ == '__main__':
    # thetas = scipy.io.loadmat('ex4weights.mat')
    matrix = scipy.io.loadmat('ex4data1.mat')
    x = matrix['X']
    y = matrix['y']
    input_size = x.shape[1]
    hidden_size = 25
    num_labels = 10
    max_iter = 1000
    lamb = 1
    alpha = 2
    theta1 = InitializeTheta(input_size, hidden_size)
    theta2 = InitializeTheta(hidden_size, num_labels)
    thetas = np.hstack((theta1.reshape((input_size+1)*hidden_size), theta2.reshape((hidden_size+1)*num_labels)))
    """sigmoid gradient """
    # sigmoid_gradient = sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
    # print(sigmoid_gradient)
    """Gradient Checking"""
    # cost, thetas_derive = J(x, y, thetas, input_size, hidden_size, num_labels, lamb)
    # thetas_nu = ComputeNumericalGradient(x, y, thetas, input_size, hidden_size, num_labels, lamb)
    # GraientCheck(thetas_nu, thetas_derive)
    """neuralnetwork traing and predict"""
    thetas_result, error_list, iter_list = GridentDescent(x,y,thetas, input_size, hidden_size, num_labels, alpha, lamb, max_iter)
    Predict(x,y,thetas_result,input_size, hidden_size, num_labels)  #  准确率： 97.72
    plt.plot(iter_list, error_list)
    plt.show()


