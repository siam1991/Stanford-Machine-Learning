import numpy as np
from sklearn import svm
import scipy.io
import matplotlib.pyplot as plt

def PlotDataLinear(X,y,w=None,b=None):
    x_positive = X[np.where(y == 1)[0]]
    x_negative = X[np.where(y == 0)[0]]
    plt.figure()
    plt.scatter(x_positive[:, 0], x_positive[:, 1], facecolors="red")
    plt.scatter(x_negative[:, 0], x_negative[:, 1], facecolors="blue")
    if w is not None and b is not None:
        xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        yp = -(w[0] * xp + b) / w[1]
        plt.plot(xp, yp, 'k-')
    plt.show()


def PlotDataRBF(X,y,classifier=None):
    x_positive = X[np.where(y == 1)[0]]
    x_negative = X[np.where(y == 0)[0]]
    plt.figure()
    plt.scatter(x_positive[:, 0], x_positive[:, 1], facecolors="red")
    plt.scatter(x_negative[:, 0], x_negative[:, 1], facecolors="blue")
    if classifier is not None:
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x2 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1, x2)
        points = np.c_[X1.ravel(), X2.ravel()]
        predict = classifier.predict(points)
        predict.shape = X1.shape
        plt.contour(X1, X2, predict, alpha=0.7, levels=[0, 1])
    plt.show()


#------------------Part 1----------------
matrix = scipy.io.loadmat('ex6data1.mat')
X = matrix["X"]
y = matrix["y"].ravel()
# PlotDataLinear(X, y)
svm_classifier = svm.SVC(C=1000.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                         tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                         decision_function_shape=None, random_state=None)
svm_classifier.fit(X, y)
w = svm_classifier.coef_[0]
b = svm_classifier.intercept_[0]
# 绘制boundary
PlotDataLinear(X, y, w, b)


#----------------Part 2------------------
matrix = scipy.io.loadmat('ex6data2.mat')
X = matrix["X"]
y = matrix["y"].ravel()
# PlotDataRBF(X, y)
svm_classifier = svm.SVC(C=1000.0, kernel='rbf', degree=3, gamma=10, coef0=0.0, shrinking=True, probability=False,
                         tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                         decision_function_shape=None, random_state=None)
svm_classifier.fit(X, y)
PlotDataRBF(X, y, svm_classifier)


#--------------Part 3--------------------
matrix = scipy.io.loadmat('ex6data3.mat')
X = matrix["X"]
y = matrix["y"].ravel()
Xval = matrix["Xval"]
yval = matrix["yval"].ravel()
vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
predict_error = np.zeros((8, 8))
for id_c, c in enumerate(vec):
    for id_gamma, gamma in enumerate(vec):
        svm_classifier = svm.SVC(C=c, kernel='rbf', gamma=gamma, shrinking=True, probability=False,
                                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                                 decision_function_shape="ovr", random_state=None)
        svm_classifier.fit(X, y)
        predict = svm_classifier.predict(Xval)
        predict_error[id_c, id_gamma] = np.mean(predict!=yval)

row_min = np.min(predict_error, axis=0)
row_id = np.argmin(predict_error,  axis=0)
col_id = np.argmin(row_min)

c_id = row_id[col_id]
gamma_id = col_id
C_suit = vec[c_id]
gamma_suit = vec[gamma_id]
print(C_suit, gamma_suit)
svm_classifier = svm.SVC(C=C_suit, kernel='rbf', gamma=gamma_suit, shrinking=True, probability=False,
                         tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                         decision_function_shape='ovr', random_state=None)
svm_classifier.fit(X, y)
PlotDataRBF(X, y, svm_classifier)


#--------------Spam Classifier-----------------------------
matrix = scipy.io.loadmat('spamTrain.mat')
X = matrix["X"]
# y为二维数组，需要将其ravel(),否则预测结果与之对比时，会出错（广播）
y = matrix["y"].ravel()
svm_classifier = svm.SVC(C=0.1, kernel='linear',tol=1e-3)
svm_classifier.fit(X, y)
predict = svm_classifier.predict(X)
Accuracy = np.mean(predict==y)
print(Accuracy*100)
test_matrix = scipy.io.loadmat('spamTest.mat')
test_X = test_matrix["Xtest"]
test_y = test_matrix["ytest"].ravel()
test_predict = svm_classifier.predict(test_X)
test_Accuracy = np.mean(test_predict==test_y)
print(test_Accuracy*100)