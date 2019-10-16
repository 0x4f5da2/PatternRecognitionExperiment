import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


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

    def fit(self, x_train):
        self.set_fit_data(x_train)
        while self.fit_iteration():
            pass

    def set_fit_data(self, x_train: np.array):
        self._x_train = x_train
        self._len_train = x_train.shape[0]
        self._dim = x_train.shape[1]
        self._alpha = np.array([1 / self._n for _ in range(self._n)])
        self._mean = np.array(
            [i / self._n * (x_train.max(axis=0) - x_train.min(axis=0)) + x_train.min(axis=0) for i in range(self._n)])
        # self._mean = np.array([[1, 2], [2, 3]])
        self._var = np.array([np.ones(self._dim) for i in range(self._n)])

    def _em_iteration(self):
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

    def fit_iteration(self):
        mu, sigma, alpha = self._em_iteration()
        delta = np.sum((self._mean - mu) * (self._mean - mu)) + np.sum(
            (self._var - sigma) * (self._var - sigma)) + np.sum((self._alpha - alpha) * (self._alpha - alpha))
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

    def score_samples(self, X):
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

    def predict(self, X):
        gi = self.score_samples(X)
        return np.argmax(gi, axis=0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    n_sample_1 = 50
    n_sample_2 = 30
    centers = [[0.0, 0.0], [3.0, 4.0]]
    clusters_std = [1, 0.2]
    X, y = make_blobs(n_samples=[n_sample_1, n_sample_2], centers=centers, cluster_std=clusters_std, random_state=0,
                      shuffle=False)
    clf = GaussianMixtureModel(2)
    clf.fit(X)
    cls = clf.predict(X)
    score = clf.score_samples(X)
    print(score)
    plt.scatter(X[:, 0], X[:, 1], c=cls, cmap=plt.cm.Paired, edgecolors='k')
    plt.show()
