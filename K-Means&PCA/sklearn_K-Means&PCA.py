from sklearn.cluster import KMeans
import scipy.io
import matplotlib.pyplot as plt


"""
KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0,
       random_state=None, copy_x=True, n_jobs=1)
Parameter:
n_clusters : int, optional, default: 8
    The number of clusters to form as well as the number of
    centroids to generate.

max_iter : int, default: 300
    Maximum number of iterations of the k-means algorithm for a
    single run.

n_init : int, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.

init : {'k-means++', 'random' or an ndarray}
    Method for initialization, defaults to 'k-means++':

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

precompute_distances : {'auto', True, False}
    Precompute distances (faster but takes more memory).

    'auto' : do not precompute distances if n_samples * n_clusters > 12
    million. This corresponds to about 100MB overhead per job using
    double precision.

    True : always precompute distances

    False : never precompute distances

tol : float, default: 1e-4
    Relative tolerance with regards to inertia to declare convergence

n_jobs : int
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    If -1 all CPUs are used. If 1 is given, no parallel computing code is
    used at all, which is useful for debugging. For n_jobs below -1,
    (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
    are used.

random_state : integer or numpy.RandomState, optional
    The generator used to initialize the centers. If an integer is
    given, it fixes the seed. Defaults to the global numpy random
    number generator.

verbose : int, default 0
    Verbosity mode.

copy_x : boolean, default True
    When pre-computing distances it is more numerically accurate to center
    the data first.  If copy_x is True, then the original data is not
    modified.  If False, the original data is modified, and put back before
    the function returns, but small numerical differences may be introduced
    by subtracting and then adding the data mean.

Attribute:

cluster_centers_ : array, [n_clusters, n_features]
    Coordinates of cluster centers

labels_ :
    Labels of each point

inertia_ : float
    Sum of distances of samples to their closest cluster center.

"""


def PlotData(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()

def Plot(X, K, centroids):
    plt.scatter(X[:, 0],X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", color='red')
    plt.show()


if __name__ == '__main__':
    # ---------part 1------------------------
    matrix = scipy.io.loadmat("ex7data2.mat")
    X = matrix["X"]
    # PlotData(X)
    K = 3
    max_iter = 10
    K_Means = KMeans(n_clusters=K, max_iter=max_iter, init='k-means++', n_init=10, tol=0.0001,
                     precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
    K_Means.fit(X)
    centroids_record = K_Means.cluster_centers_
    # print(centroids_record)
    Plot(X, K, centroids_record)
    # --------part 2----------------------------
    # picture = scipy.io.loadmat("bird_small.mat")
    # img = picture["A"]/255
    img = plt.imread("bird_small.png")
    img_size = img.shape
    X = img.reshape(img_size[0]*img_size[1], 3)
    K = 16
    max_iter = 10
    K_Means = KMeans(n_clusters=K, max_iter=max_iter,init='k-means++', n_init=10, tol=0.0001,
                     precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
    K_Means.fit(X)
    centroids = K_Means.cluster_centers_
    labels = K_Means.labels_
    X_trans = centroids[labels, :]
    X_trans = X_trans.reshape(img_size[0], img_size[1],3)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(X_trans)
    plt.show()
