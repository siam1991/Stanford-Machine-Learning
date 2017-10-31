import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Collaborative Filtering 协同过滤

"""
根据
Y: num_movies*num_users 存有用户对电影的打分
R: num_movies*num_users 值为0或1，1表示用户对该部电影打分，0表示没有
求出
X: num_movies*num_movie_feature,表示电影的特征
Theta:num_users*num_movie_feature,表示用户对各类型电影的喜爱程度
"""

def NormalizeRating(Y, R):
    # 根据每部电影的平均得分来scale
    m = Y.shape[0]
    Y_norm = np.zeros_like(Y)
    Y_mean =- np.zeros((m,1))
    for i in range(m):
        ind = np.where(R[i,:]==1)
        Y_mean[i,0] = np.mean(Y[i,ind])
        Y_norm[i,ind] = Y[i,ind]-Y_mean[i, 0]
    return Y_mean, Y_norm


def LoadMovie(filename):
    movie_list =[]
    with open(filename,encoding="utf-8") as movie_file:
        for line in movie_file:
            movie_list.append(' '.join(line.split()[1:]))
    return movie_list


def CostFunc(X, Theta, Y, R, lamb):
    ERROR = np.dot(X, Theta.T)-Y
    J = 1/2*np.sum(R*(ERROR**2))+lamb/2*np.sum(Theta**2) + lamb/2*np.sum(X**2)
    X_grad = np.dot(R*ERROR, Theta)+lamb*X
    Theta_grad = np.dot((R*ERROR).T, X)+lamb*Theta
    return J, X_grad, Theta_grad


def GradientDescent(X,Theta,Y,R,lamb,alpha, max_iter):
    iter_list = []
    cost_list = []
    for i in range(max_iter):
        J,x_grad,theta_grad = CostFunc(X,Theta,Y,R,lamb)
        X = X-alpha*x_grad
        Theta = Theta-alpha*theta_grad
        iter_list.append(i)
        cost_list.append(J)
    # plt.plot(iter_list, cost_list)
    # plt.show()
    return X, Theta




if __name__ == '__main__':
    movie_matrix = scipy.io.loadmat("ex8_movies.mat")
    Y = movie_matrix["Y"]
    R = movie_matrix["R"]
    param = scipy.io.loadmat("ex8_movieParams.mat")
    X = param["X"]
    Theta = param["Theta"]
    # plt.imshow(Y)
    # plt.show()
    # test costfunc
    # num_users = 4
    # num_movies = 5
    # num_features = 3
    # X = X[0:num_movies, 0:num_features]
    # Theta = Theta[0:num_users, 0:num_features]
    # Y = Y[0:num_movies, 0:num_users]
    # R = R[0:num_movies, 0:num_users]
    # J, X_grad, Theta_grad = CostFunc(X, Theta, Y, R, 0)
    # print(J)
    # J, X_grad, Theta_grad = CostFunc(X, Theta, Y, R, 1.5)
    # print(J)
    movie_list = LoadMovie("movie_ids.txt")
    my_ratings = np.zeros([1682, 1])
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    my_r = np.where(my_ratings!=0, 1, 0)
    Y = np.hstack((my_ratings, Y))
    R = np.hstack((my_r, R))
    Y_mean, Y_norm = NormalizeRating(Y,R)
    num_movies,num_users = Y.shape
    num_features = 10
    lamb = 10
    alpha = 0.001
    max_iter = 500
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)
    X, Theta = GradientDescent(X, Theta, Y, R, lamb, alpha, max_iter)
    predicts = np.dot(X, Theta.T)

    my_predict = predicts[:, 0].reshape(-1, 1)+Y_mean
    my_predict = my_predict.reshape(-1)
    print("************recommender for you*****************")
    """
    # Predicting rating 9.0 for movie Titanic (1997)
    # Predicting rating 8.9 for movie Star Wars (1977)
    # Predicting rating 8.8 for movie Shawshank Redemption, The (1994)
    # Predicting rating 8.5 for movie Good Will Hunting (1997)
    # Predicting rating 8.5 for movie Usual Suspects, The (1995)
    # Predicting rating 8.4 for movie Raiders of the Lost Ark (1981)
    # Predicting rating 8.4 for movie Empire Strikes Back, The (1980)
    # Predicting rating 8.4 for movie Braveheart (1995)
    # Predicting rating 8.5 for movie As Good As It Gets (1997)
    """
    idx = np.argsort(my_predict)[::-1]

    for i in idx[:10]:
        print("Predicting rating {} for movie {}".format(my_predict[i], movie_list[i]))
    print("***********original****************************")
    for i in range(my_ratings.shape[0]):
        if my_ratings[i,0]>0:
            print(movie_list[i])








