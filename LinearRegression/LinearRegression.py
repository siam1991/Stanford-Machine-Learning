import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 寻找合适的theta参数，使得J的值最小
class LinearRegression(object):
    """
    根据已有的训练集，求取合适的theta
    """
    def __init__(self, x, z, alpha, maxstep, method=1):
        """
        x: 特征矩阵 m*n，未包含x0
        z: 结果矩阵 m*1
        alpha: 学习速率
        maxstep: 最大迭代步
        method: 求解方法，method=1，梯度下降法，method=0，normal equation,默认梯度下降
        """
        self.method = method
        self.x = x
        self.sample_size, self.feature_size = x.shape
        self.z = z.reshape(self.sample_size, 1)
        self.maxstep = maxstep
        self.theta = np.zeros((self.feature_size+1, 1))
        self.alpha = alpha  # 学习速率
        self.error_record = []    # 记录误差
        self.iter_record = []   # 记录迭代步数

    def Predict(self, predict):
        predict_sample, predict_feature = predict.shape
        if self.method:
            # 梯度下降法，得到系数
            self.GridentDescent()
            plt.plot(self.iter_record, self.error_record)
            plt.show()
            if predict_feature >= 2:
                predict = (predict - self.mu)/self.sigma
            predict = np.hstack(([[1]]*predict_sample, predict))
            result = np.dot(predict, self.theta)  # 预测结果

        else:
            # normal equation , 得到系数
            self.NormalizeEquation()
            predict = np.hstack(([[1]]*predict_sample, predict))
            result = np.dot(predict, self.theta)  # 预测结果
        return result

    def J_theta(self):
        """
        cost function
        theta: 选取的参数矩阵
        x:输入矩阵，训练集的特征数据
        :return: cost function 的值
        """
        return 0.5/self.sample_size*np.sum((np.dot(self.x, self.theta)-self.z)**2)

    def GridentDescent(self):
        """
        归一化 只有梯度下降法才需要特征归一化,归一化之后再添加x0=1那一列
        特征数大于1时，如果尺度差别较大，才需要进行归一
        :return:
        """
        self.FeatureScale()
        self.x = np.hstack(([[1]] * self.sample_size, self.x))
        count = 0
        while count <= self.maxstep:
            derive = np.dot(self.x.T, (np.dot(self.x, self.theta)-self.z))
            self.theta = self.theta-self.alpha/self.sample_size*derive
            error_init = self.J_theta()
            count += 1
            self.iter_record.append(count)
            self.error_record.append(error_init)
        print("Grident descent theta:", self.theta)


    def NormalizeEquation(self):
        self.x = np.hstack(([[1]] * self.sample_size, self.x))
        inverse = np.linalg.inv(np.dot(self.x.T, self.x))
        self.theta = np.dot(inverse, np.dot(self.x.T, self.z))
        print("Normal Equation theta:", self.theta)


    def FeatureScale(self):
        """
        将输入矩阵归一化
        线型归一化：single_value-min/(max-min)
            diff = np.max(x,axis=0)-np.min(x,axis=0)
            return (x-np.min(x, axis=0))/diff
            return self.x
        均值标准化：single_value-
        :return:
        """
        if self.feature_size >= 2:
            self.mu = np.array(np.mean(self.x, axis=0))
            self.sigma = np.array(np.std(self.x, axis=0))
            self.x = (self.x-self.mu)/self.sigma

dataframe = pd.read_csv('ex1data2.txt', header=None)
columns = dataframe.shape[1]-1
x = dataframe[list(range(columns))]
z = dataframe[columns]


if __name__ == '__main__':
    alpha = 0.01
    maxstep = 400
    method = 1
    linear = LinearRegression(x, z, alpha, maxstep, method)
    result = linear.Predict(np.array([[1650, 3]]))
    print(result)

