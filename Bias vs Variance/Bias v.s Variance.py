import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def polyFeatures(x, p):
    sample_size = x.shape[0]
    x_poly = np.zeros((sample_size, p))
    for i in range(p):
        x_poly[:,i] = x[:,0]**(i+1)
    return x_poly


def featureNormalize(x):
    mu = np.array(np.mean(x, axis=0))
    sigma = np.array(np.std(x, axis=0))
    x = (x-mu)/sigma
    return x, mu, sigma


def costFunction(x,y,theta,lamb):
    sample_size = x.shape[0]
    theta_no = np.vstack(([[0]], theta[1:]))
    J = 1/2/sample_size*np.sum((np.dot(x, theta)-y)**2)+lamb/2/sample_size*np.sum(theta_no**2)
    grad = 1/sample_size*np.dot(x.T,(np.dot(x,theta)-y))+lamb/sample_size*theta_no
    return J, grad


def gradientDescent(x,y,lamb,alpha,max_iter):
    feature_size = x.shape[1]
    sample_size = x.shape[0]
    J_set = np.zeros((max_iter, 1))
    theta = np.zeros((feature_size,1))
    for i in range(max_iter):
        J, grad = costFunction(x,y,theta, lamb)
        theta = theta-alpha/sample_size*grad
        J_set[i,0] = J
    plt.plot(range(max_iter), J_set)
    plt.show()
    return theta


def plotLearningCurve(x,y,xval,yval,lamb_train,alpha,max_iter):
    train_error = np.zeros_like(y)
    val_error = np.zeros_like(y)
    sample_size = x.shape[0]
    for i in range(1, sample_size+1):
        x_train = x[0:i, :]
        y_train = y[0:i, :]
        """经过测试，随着样本数量的增大，学习速率也可以增大，
        若采用相同的学习速率，则在后期需要较大的迭代步数"""
        theta = gradientDescent(x_train, y_train, lamb_train, alpha*i, max_iter)
        train_error[i-1], grad = costFunction(x_train, y_train, theta, 0)
        val_error[i-1], grad = costFunction(xval, yval, theta, 0)
    plt.plot(range(1, sample_size+1), train_error)
    plt.plot(range(1, sample_size+1), val_error)
    plt.show()


def validationCurve(x,y,xval,yval,alpha,max_iter):
    lambda_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,3, 10]
    train_error = np.zeros((len(lambda_list),1))
    val_error = np.zeros((len(lambda_list),1))
    for i, lamb in enumerate(lambda_list):
        theta = gradientDescent(x, y, lamb, alpha, max_iter)
        train_error[i], grad = costFunction(x, y, theta,0)
        val_error[i], grad = costFunction(xval, yval, theta, 0)
    plt.plot(lambda_list, train_error)
    plt.plot(lambda_list, val_error)
    plt.show()


if __name__ == '__main__':
    # 训练集
    matrix = scipy.io.loadmat('ex5data1.mat')
    # 训练集
    x = matrix['X']
    y = matrix['y']
    # 交叉验证集
    xval = matrix['Xval']
    yval = matrix['yval']
    """part1：test costfunc and plot learning curve"""
    # 增加特征0
    x = np.hstack(([[1]] * x.shape[0], x))
    xval = np.hstack(([[1]] * xval.shape[0], xval))
    theta = np.ones((x.shape[1],1))
    J, grad = costFunction(x, y, theta, 1)
    print(J)
    lamb_train = 0
    alpha = 0.001
    max_iter = 3000
    # learning curve
    plotLearningCurve(x, y, xval, yval, lamb_train, alpha,max_iter)
    """part2: poly and different lambda"""
    # lamb_train = 0
    # alpha = 0.001
    # max_iter = 3000
    # x = polyFeatures(x, 8)
    # x, mu, sigma = featureNormalize(x)
    # x = np.hstack(([[1]] * x.shape[0], x))
    # xval = polyFeatures(xval, 8)
    # xval = (xval-mu)/sigma
    # xval = np.hstack(([[1]] * xval.shape[0], xval))
    # # poly feature learning curve
    # plotLearningCurve(x, y, xval, yval, lamb_train, alpha, max_iter)
    # # validation curve
    # alpha = 0.3
    # max_iter = 400
    # validationCurve(x, y, xval, yval, alpha, max_iter)


