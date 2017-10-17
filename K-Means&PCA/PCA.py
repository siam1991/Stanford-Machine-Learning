import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.io



def PlotData(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

def FeatureNormalize(X):
    mu = np.array(np.mean(X, axis=0))
    sigma = np.array(np.std(X, axis=0))
    X = (X- mu) / sigma
    return mu,sigma, X

def PCA(X):
    m,n = X.shape
    Sigma = 1/m*np.dot(X.T, X)
    U,S,V=linalg.svd(Sigma)
    return U, S

def ProjectData(X,U,K):
    U_reduce = U[:,0:K]
    Z = np.dot(X,U_reduce)
    return Z

def RecoverData(Z, U,K):
    X_recover = np.dot(Z,U[:,0:K].T)
    return X_recover


if __name__ == '__main__':
    matrix = scipy.io.loadmat("ex7data1.mat")
    X = matrix["X"]
    PlotData(X)
    mu,sigma,X = FeatureNormalize(X)
    k=1
    U,S = PCA(X)
    print(U[0:2,0])
    #------------------
    # PlotData(X)
    Z=ProjectData(X,U,k)
    print(Z[0,0])
    X_rec = RecoverData(Z,U,k)
    print(X_rec[0,:])
    #-----------------------
    face = scipy.io.loadmat("ex7faces.mat")
    X_face = face["X"]
    plt.imshow(X_face[0:100, :]/255)
    plt.show()
    mu,sigma,X_face = FeatureNormalize(X_face)
    U_face,S_face = PCA(X_face)
    k_face = 100
    Z_face = ProjectData(X_face,U_face,k_face)
    print(Z_face.shape)
    X_face_rec = RecoverData(Z_face,U_face,k_face)
    plt.imshow(X_face_rec[0:100, :])
    plt.show()

