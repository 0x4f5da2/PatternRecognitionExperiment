import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB


SPLITTING_RATIO = 0.8

MESH_EPS = 0.05


class NaiveBayes:
    def __init__(self, lambda_=1):
        self._lambda_ = lambda_
        self._prior = dict()
        self._cls = None
        self._mean = dict()
        self._cov = dict()
        self._dim = None
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        self._dim = X.shape[1]
        self._cls = np.unique(y)
        for each in self._cls:
            self._prior[each] = (y[y == each].shape[0] + self._lambda_) / (
                    y.shape[0] + self._cls.shape[0] * self._lambda_)
            self._mean[each] = np.mean(X[y == each], axis=0)
            self._cov[each] = np.cov(X[y == each].T) * np.eye(self._dim)  # 假设独立同分布
        self._fitted = True

    def calc_gi(self, X):
        if not self._fitted:
            raise RecursionError
        gi_list = []
        for each in self._cls:
            result = -0.5 * np.sum(
                (X - self._mean[each]) @ np.linalg.inv(self._cov[each]) * (X - self._mean[each]),
                axis=1) \
                     - 1 / 2 * np.log(np.linalg.det(self._cov[each])) \
                     + np.log(self._prior[each])
            gi_list.append(result)
        gi = np.vstack(gi_list)
        return gi

    def predict(self, X):
        gi = self.calc_gi(X)
        return self._cls[np.argmax(gi, axis=0)]

    def get_mean(self):
        return self._mean

    def get_cov(self):
        return self._cov


if __name__ == '__main__':
    # 读入数据
    with open("boy.txt") as f:
        boy_data = [list(map(float, each.strip().split())) for each in f.readlines()]
    with open("girl.txt") as f:
        girl_data = [list(map(float, each.strip().split())) for each in f.readlines()]
    # 转化为numpy格式并打乱
    boy_data = np.array(boy_data)
    boy_data = boy_data[:, :2]
    girl_data = np.array(girl_data)
    girl_data = girl_data[:, :2]
    # shuffle(boy_data)
    # shuffle(girl_data)
    # 计算决策边界相关
    data = np.vstack([boy_data, girl_data])
    xx, yy = np.meshgrid(np.arange(data.min(axis=0)[0] - 5, data.max(axis=0)[0] + 5, MESH_EPS),
                         np.arange(data.min(axis=0)[1] - 5, data.max(axis=0)[1] + 5, MESH_EPS))
    # 划分训练集以及测试集
    boy_train_data = boy_data[:int(SPLITTING_RATIO * boy_data.shape[0])]
    boy_test_data = boy_data[int(SPLITTING_RATIO * boy_data.shape[0]):]
    girl_train_data = girl_data[:int(SPLITTING_RATIO * girl_data.shape[0])]
    girl_test_data = girl_data[int(SPLITTING_RATIO * girl_data.shape[0]):]
    # 准备训练数据以及测试数据
    train_x = np.vstack([boy_train_data, girl_train_data])
    train_y = np.hstack([
        np.ones(boy_train_data.shape[0]).astype(np.int) * -1,
        np.ones(girl_train_data.shape[0]).astype(np.int)
    ])
    test_x = np.vstack([boy_test_data, girl_test_data])
    test_y = np.hstack([
        np.ones(boy_test_data.shape[0]).astype(np.int) * -1,
        np.ones(girl_test_data.shape[0]).astype(np.int)
    ])
    # 开始计算
    start = time.time()
    bayes = NaiveBayes(0)
    bayes.fit(train_x, train_y)
    predicted = bayes.predict(test_x)
    print("自行编写的朴素贝叶斯分类器的正确率：")
    print(np.sum(predicted == test_y) / predicted.shape[0])
    print("自行编写的朴素贝叶斯分类器的用时：")
    print(time.time() - start)

    # 绘制决策边界
    sk = GaussianNB()
    sk.fit(train_x, train_y)
    Z_sk = sk.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
    Z_sk = Z_sk.reshape(xx.shape)

    Z = bayes.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    figure = plt.figure(figsize=(36, 9))
    ax = plt.subplot(1, 2, 1)
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(train_x[train_y == -1, 0], train_x[train_y == -1, 1], marker="x", c='b', alpha=0.5)
    ax.scatter(train_x[train_y == 1, 0], train_x[train_y == 1, 1], marker="o", c='r', alpha=0.5)
    ax = plt.subplot(1, 2, 2)
    ax.contourf(xx, yy, Z_sk, alpha=0.8)
    ax.scatter(train_x[train_y == -1, 0], train_x[train_y == -1, 1], marker="x", c='b', alpha=0.5)
    ax.scatter(train_x[train_y == 1, 0], train_x[train_y == 1, 1], marker="o", c='r', alpha=0.5)
    plt.show()
