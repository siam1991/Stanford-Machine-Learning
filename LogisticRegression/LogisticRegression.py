import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def MapFeature(x1, x2):
    x = np.ones((x1.shape[0], 27))
    degree = 6
    count =0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            x[:, count] = x1 ** (i - j) * x2 ** j
            count += 1
    return x


class LogisticRegression(object):
    def __init__(self, x, z, alpha, maxstep,lam=10):
        self.x1 = np.array(x[0])
        self.x2 = np.array(x[1])
        self.x = x
        self.origin_x= x
        self.sample_size, self.feature_size = self.x.shape
        self.z = z.reshape(self.sample_size, 1)
        self.alpha = alpha
        self.maxstep = maxstep
        self.theta = np.zeros((self.feature_size+1,1))
        self.lam = lam
        self.iter = []
        self.error = []

    def H(self):
        return 1 / (1.0 + np.exp(-np.dot(self.x, self.theta)))

    def J(self):
        exp_part = np.dot(self.x, self.theta)  # m*1
        part1 = np.log(1.0+np.exp(exp_part))
        part2 = np.dot(self.z.T, np.dot(self.x, self.theta))
        return 1.0/self.sample_size*(np.sum(part1)-part2)

    def GridentDescent(self):
        self.ScaleFeature()
        self.x = np.hstack(([[1]]*self.sample_size, self.x))
        count = 0
        while count<self.maxstep:
            count +=1
            J_cost = self.J()
            H = self.H()
            self.theta = self.theta-self.alpha/self.sample_size*np.dot(self.x.T,(H-self.z))
            self.iter.append(count)
            self.error.append(J_cost[0])
        print(self.theta)
        return self.theta


    def J_Reg(self):
        exp_part = np.dot(self.x, self.theta)  # m*1
        part1 = np.log(1.0 + np.exp(exp_part))
        part2 = np.dot(self.z.T, exp_part)
        Regualize = np.vstack((np.array([0.0]), self.theta[1:]))**2
        return 1.0 / self.sample_size * (np.sum(part1) - part2)+self.lam/2/self.sample_size*np.sum(Regualize)


    def GridentDescent_Reg(self):
        self.ScaleFeature()
        self.x = np.hstack(([[1]] * self.sample_size, self.x))
        count = 0
        while count < self.maxstep:
            count += 1
            J_cost = self.J_Reg()
            H = self.H()
            reg = 1/self.sample_size * np.dot(self.x.T, (H - self.z))+self.lam/self.sample_size*np.vstack((np.array([0.0]),self.theta[1:]))
            self.theta = self.theta - self.alpha*reg
            self.iter.append(count)
            self.error.append(J_cost[0])
        print(self.theta)
        return self.theta

    def Predict(self, predict):
        predict_sample, predict_feature = predict.shape
        # # 梯度下降法，得到系数
        # self.GridentDescent()
        # 正则化
        self.GridentDescent_Reg()
        plt.plot(self.iter, self.error)
        plt.show()
        if predict_feature >= 2:
            predict = (predict - self.mu) / self.sigma
        predict = np.hstack(([[1]] * predict_sample, predict))
        # #  预测结果
        result = 1/(1+np.exp(-np.dot(predict, self.theta)))
        result = np.where(result > 0.5, 1, 0)
        # #  模型准确率
        p = np.dot(self.x, self.theta)
        p = np.where(p >= 0.5, 1, 0)
        correct_percent = np.sum(p == self.z) / self.sample_size * 100  # np.mean(p==self.z)*100
        print("Train Accuracy:", correct_percent)
        return result

    def ScaleFeature(self):
        if self.feature_size >= 2:
            self.mu = np.array(np.mean(self.x, axis=0))
            self.sigma = np.array(np.std(self.x, axis=0))
            self.x = (self.x - self.mu) / self.sigma

#  使用pandas读取文件后，使用numpy转换一下
dataframe = pd.read_csv('ex2data2.txt', header=None)
columns = dataframe.shape[1]-1
x = dataframe[list(range(columns))]
z = dataframe[columns]
x1 = np.array(x[0])
x2 = np.array(x[1])
plt.scatter(x1[np.where(z == 0)], x2[np.where(z == 0)], marker='o', c='red', s=50)
plt.scatter(x1[np.where(z == 1)], x2[np.where(z == 1)], marker='+', c='blue', s=50)
plt.show()

if __name__ == '__main__':
    # ex1
    # ex2data1  target answer: 0.775 + / - 0.00245, 85
    # alpha = 0.5
    # maxstep = 1500
    # logistic = LogisticRegression(x, z, alpha, maxstep)
    # result = logistic.Predict(np.array([[45, 85]]))
    # print('预测结果：', result)
    # ex2
    alpha = 0.8
    maxstep = 200
    x = MapFeature(x1, x2)
    logistic = LogisticRegression(x, z, alpha, maxstep)
    predict = MapFeature(np.array([0.13191]), np.array([-0.51389]))
    result = logistic.Predict(predict)
    print('预测结果：', result)