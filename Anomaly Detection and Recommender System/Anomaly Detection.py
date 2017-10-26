import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def VisualizeData(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()


def EstimateGaussian(X):
    mu = np.array(np.mean(X, axis=0))
    sigma = np.array(np.mean((X-mu)**2,axis=0))
    return mu, sigma

def PlotContourLine(X, mu,sigma):
    x1 = np.arange(0, 35, 0.5)
    x2 = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.c_[X1.ravel(), X2.ravel()]
    p = MultivariateGaussian(points, mu, sigma)
    p.shape = X1.shape
    plt.scatter(X[:,0],X[:,1])
    plt.contour(X1, X2, p, 10.**np.arange(-20, 0, 3), alpha=0.7)
    # plt.show()


def MultivariateGaussian(X, mu, sigma):
    k = len(sigma)
    # 将其转换为方形矩阵

    sigma = np.diag(sigma)
    X = X-mu
    p = (2*np.pi)**(-k/2)*np.linalg.det(sigma)**(-0.5)*np.exp(-0.5*np.sum(np.dot(X, np.linalg.inv(sigma))*X,axis=1))
    return p

def SelectThreshold(yval,pval):
    epsilon_best = 0
    F1_best = 0
    stepsize = (np.max(pval)-np.min(pval))/1000
    epsilon_list = np.arange(0, np.max(pval)+stepsize, stepsize)
    for epsilon in epsilon_list:
        # print(epsilon)
        predictions = np.where(pval < epsilon, 1, 0)
        predictions = predictions.reshape(-1, 1)
        tp = np.sum(np.logical_and(yval == 1, predictions == 1))
        fp = np.sum(np.logical_and(yval == 0, predictions == 1))
        fn = np.sum(np.logical_and(yval == 1, predictions == 0))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > F1_best:
            F1_best = F1
            epsilon_best = epsilon
    return F1_best, epsilon_best


if __name__ == '__main__':
    matrix = scipy.io.loadmat("ex8data1.mat")
    X = matrix["X"]
    Xval = matrix["Xval"]
    yval = matrix["yval"]
    # VisualizeData(X)
    mu,sigma = EstimateGaussian(X)
    p = MultivariateGaussian(X,mu,sigma)
    PlotContourLine(X, mu, sigma)
    pval = MultivariateGaussian(Xval,mu,sigma)
    F1,epsilon = SelectThreshold(yval,pval)
    print(F1)
    print(epsilon)
    anomaly_index = np.argwhere(p<epsilon)
    print(anomaly_index)
    plt.scatter(X[anomaly_index,0],X[anomaly_index,1],edgecolors='red')
    plt.show()