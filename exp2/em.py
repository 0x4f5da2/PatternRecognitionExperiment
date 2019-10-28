import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


# 使用numpy向量化计算，运算效率高，代码简洁易读
# 将EM算法的过程可视化

class GaussianMixtureModel:
    def __init__(self, n=1, eps=5e-4):
        self._n = n
        self._alpha = None
        self._mean = None
        self._var = None
        self._dim = None
        self._fitted = False
        self._x_train = None
        self._len_train = None
        self._eps = eps

    def fit(self, x_train):  # 执行EM直到收敛
        self.set_fit_data(x_train)
        while self.fit_iteration():
            pass

    def set_fit_data(self, x_train: np.array):  # 设置训练数据
        self._x_train = x_train
        self._len_train = x_train.shape[0]
        self._dim = x_train.shape[1]
        self._alpha = np.array([1 / self._n for _ in range(self._n)])
        self._mean = np.array(
            [i / self._n * (x_train.max(axis=0) - x_train.min(axis=0)) + x_train.min(axis=0) for i in range(self._n)])
        self._var = np.array([np.ones(self._dim) for i in range(self._n)])

    def _em_iteration(self):  # 实现统计学习方法9.3节内容，EM算法主要部分，使用向量化计算
        self._fitted = True
        # Expectation
        results = []
        for k in range(self._n):
            cov = np.eye(self._dim) * self._var[k] + np.eye(self._dim) * 1e-6
            result = -0.5 * np.sum(
                (self._x_train - self._mean[k]) @ np.linalg.inv(cov) * (self._x_train - self._mean[k]), axis=1
            ) \
                     - 1 / 2 * np.log(np.linalg.det(cov)) \
                     - self._dim / 2 * np.log(2 * np.pi)
            result = np.exp(result)
            result = self._alpha[k] * result
            results.append(result)
        gamma_hat = np.vstack(results).T
        gamma_hat = gamma_hat / np.sum(gamma_hat, axis=1)[..., None]
        # Maximization
        mu_hat = gamma_hat.T @ self._x_train / (np.sum(gamma_hat, axis=0)[..., None])
        sigma2_hat = np.sum(np.exp2(self._x_train[..., None] - mu_hat.T) * gamma_hat[:, None, :], axis=0).T / np.sum(
            gamma_hat, axis=0)[..., None]
        alpha_hat = np.sum(gamma_hat, axis=0) / self._len_train

        return mu_hat, sigma2_hat, alpha_hat

    def fit_iteration(self):  # 执行一次EM算法，返回是否继续迭代
        mu, sigma, alpha = self._em_iteration()
        delta = np.sum(np.power(self._mean - mu, 2)) + np.sum(np.power(self._var - sigma, 2)) + np.sum(
            np.power(self._alpha - alpha, 2))
        self._mean = mu
        self._var = sigma
        self._alpha = alpha
        logging.info("delta: {}".format(delta))
        logging.info("mean: {}".format(str(self._mean)))
        logging.info("sigma: {}".format(str(self._var)))
        logging.info("alpha: {}".format(str(self._alpha)))
        logging.info("...................")
        if delta < self._eps:
            return False
        else:
            return True

    def score_samples(self, X):  # 获取每个高斯分布的概率密度
        if not self._fitted:
            raise RuntimeError
        score_list = []
        for k in range(self._n):
            cov = np.eye(self._dim, self._dim) * self._var[k] + 1e-6
            result = -0.5 * np.sum(
                (X - self._mean[k]) @ np.linalg.inv(cov) * (X - self._mean[k]),
                axis=1) \
                     - 1 / 2 * np.log(np.linalg.det(cov)) \
                     - self._dim / 2 * np.log(2 * np.pi)
            result = np.exp(result)
            result = self._alpha[k] * result
            score_list.append(result)
        score = np.vstack(score_list)
        return score

    def predict(self, X):  # 预测
        gi = self.score_samples(X)
        return np.argmax(gi, axis=0)


if __name__ == '__main__':
    MESH_EPS = 0.05
    logging.basicConfig(level=logging.INFO)
    n_sample_1 = 90
    n_sample_2 = 80
    centers = [[0.0, 6.0], [6.0, 0]]
    clusters_std = [2, 2]
    # 生成数据
    X, y = make_blobs(n_samples=[n_sample_1, n_sample_2], centers=centers, cluster_std=clusters_std, random_state=0,
                      shuffle=False)
    # 实例化所实现算法
    clf = GaussianMixtureModel(2)
    clf.set_fit_data(X)
    seq = 0
    xx, yy = np.meshgrid(np.arange(X.min(axis=0)[0] - 3, X.max(axis=0)[0] + 3, MESH_EPS),
                         np.arange(X.min(axis=0)[1] - 3, X.max(axis=0)[1] + 3, MESH_EPS))
    # 进行迭代
    while True:
        c = clf.fit_iteration()
        # 绘制此次迭代后的结果
        scores = clf.score_samples(np.vstack([xx.ravel(), yy.ravel()]).T)
        scores = scores[1, :] - scores[0, :]
        scores = scores.reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        CS = plt.contour(xx, yy, scores, alpha=0.8)
        plt.clabel(CS, CS.levels, inline=True, fontsize=10)
        plt.savefig("./{}.png".format(seq))
        plt.show()
        plt.close()
        seq += 1
        if not c:
            break
    # 生成可视化GIF
    import os
    os.system("rm *.gif")
    os.system("ffmpeg -framerate 5 -i %d.png  em.gif")
    os.system("rm *.png")
