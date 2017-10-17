import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def FindClosestCentroids(X, initial_centroids):
    # 聚类个数
    K = initial_centroids.shape[0]
    # ids = np.zeros(X.shape[0],1)
    distance_min = np.zeros((X.shape[0], K))
    for i, centroid in enumerate(initial_centroids):
        centroids = np.tile(centroid, (X.shape[0],1))
        distance_sqrt = np.sum((X-centroids)**2, axis=1)
        distance_min[:, i] = distance_sqrt
    ids = np.argmin(distance_min, axis=1)
    return ids

def ComputeCentroids(X, centroids, ids):
    K = centroids.shape[0]
    for i in range(K):
        centroids[i, :] = np.mean(X[ids==i, :], axis=0)
    return centroids

def InitCentriods(X, K):
    # centriods = np.zeros(K, X.shape[1])
    randidx = np.random.permutation(X.shape[0])
    centriods = X[randidx[0:K], :]
    return centriods

def K_Means(X, K, iter_max):
    centroids_log = np.zeros((iter_max+1, K, X.shape[1]))
    centriods = InitCentriods(X, K)  # np.array([[3.,3.],[6.,2.],[8.,5.]])  ##
    centroids_log[0, :, :] = centriods
    for i in range(iter_max):
        ids = FindClosestCentroids(X, centriods)
        centriods = ComputeCentroids(X, centriods, ids)
        # print(centriods)
        centroids_log[i + 1, :, :] = centriods
    # print(centriods)
    return centroids_log


def PlotData(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()

def Plot(X, K, centroids_record):
    plt.scatter(X[:, 0],X[:, 1])
    for i in range(K):
        centroid_i = centroids_record[:, i, :]
        plt.plot(centroid_i[:, 0], centroid_i[:, 1], marker="*",color='red')
    plt.show()



if __name__ == '__main__':
    #---------part 1------------------------
    matrix = scipy.io.loadmat("ex7data2.mat")
    X = matrix["X"]
    # PlotData(X)
    K = 3
    max_iter = 10
    centroids_record = K_Means(X, K, max_iter)
    Plot(X, K, centroids_record)
    #--------part 2----------------------------
    # picture = scipy.io.loadmat("bird_small.mat")
    # img = picture["A"]/255
    img = plt.imread("bird_small.png")
    img_size = img.shape
    X = img.reshape(img_size[0]*img_size[1], 3)
    K = 16
    max_iter = 10
    centroids_record = K_Means(X, K, max_iter)
    centroids = centroids_record[-1, :, :].reshape(K, 3)
    ids = FindClosestCentroids(X, centroids)
    X_trans = centroids[ids, :]
    X_trans = X_trans.reshape(img_size[0], img_size[1],3)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(X_trans)
    plt.show()




