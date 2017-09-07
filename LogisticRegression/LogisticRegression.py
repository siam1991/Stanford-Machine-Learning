import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, x,z,alpha,maxstep):
        self.x = x
        self.sample_size, self.feature_size = self.x.shape
        self.z = self.z = z.reshape(self.sample_size, 1)
        self.alpha = alpha
        self.maxstep = maxstep
        self.theta = np.zeros((self.feature_size+1,1))
        self.iter = []
        self.error = []


    def J(self):
        exp_part = np.dot(self.x, self.theta)  # m*1
        part1 = np.log(1.0+np.exp(exp_part))
        part2 = np.dot(self.z.T, np.dot(self.x, self.theta))
        return 1.0/self.sample_size*(np.sum(part1)-part2)

    def H(self):
        return 1/(1.0+np.exp(-np.dot(self.x, self.theta)))


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

    def Predict(self,predict):
        predict_sample, predict_feature = predict.shape
        # 梯度下降法，得到系数
        self.GridentDescent()
        plt.plot(self.iter, self.error)
        plt.show()
        if predict_feature >= 2:
            predict = (predict - self.mu) / self.sigma
        predict = np.hstack(([[1]] * predict_sample, predict))
        # 预测结果
        result = 1/(1+np.exp(-np.dot(predict, self.theta)))
        return result

    def ScaleFeature(self):
        if self.feature_size >= 2:
            self.mu = np.array(np.mean(self.x, axis=0))
            self.sigma = np.array(np.std(self.x, axis=0))
            self.x = (self.x - self.mu) / self.sigma


# 使用pandas读取文件后，使用numpy转换一下
dataframe = pd.read_csv('ex2data1.txt', header=None)
columns = dataframe.shape[1]-1
x = dataframe[list(range(columns))]
z = dataframe[columns]


if __name__ == '__main__':
    # ex2data1  target answer: 0.775 + / - 0.00245, 85
    alpha = 0.5
    maxstep = 1500
    logistic = LogisticRegression(x, z, alpha, maxstep)
    result = logistic.Predict(np.array([[55.48216114069585, 35.57070347228866]]))
    print(result)
    # 根据概率，确定分类
    result = np.where(result > 0.5, 1, 0)
    print(result)

