import numpy as np
import time
from sklearn import datasets
from sklearn.svm import SVC


class LinearSVM:
    def __init__(self):
        self.W = None

    def fit(self, X, y, learning_rate=1e-5, reg=1e-5, num_iter=None, batch_size=None, verbose=False):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        num_train, dim = X.shape
        num_classes = np.unique(y).shape[0]
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        for it in range(num_iter):
            if batch_size is not None:
                batch_idx = np.random.choice(num_train, batch_size, replace=True)
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
            else:
                X_batch = X
                y_batch = y

            loss, grad = self._loss(X_batch, y_batch, reg)

            self.W -= learning_rate * grad

            if verbose and it % 50 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iter, loss))

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        scores = X @ self.W
        return np.argmax(scores, axis=1)

    def _loss(self, X_batch, y_batch, reg):
        # forward
        loss = 0.0
        num_train = X_batch.shape[0]
        scores = X_batch @ self.W
        correct_class_score = scores[range(num_train), y_batch]
        margins = scores.copy() - correct_class_score[:, None] + 1
        mask = margins > 0
        mask[range(num_train), y_batch] = False
        loss += np.sum(margins[mask])
        loss /= num_train
        loss += reg * np.sum(self.W * self.W)

        # backward
        d_margin = np.zeros_like(margins)
        d_margin[mask] = 1
        d_correct_score = np.sum(d_margin, axis=1)
        d_score = np.zeros_like(margins)
        d_score[range(num_train), y_batch] -= d_correct_score
        d_score += d_margin
        dW = np.zeros(self.W.shape)
        dW += X_batch.T @ d_score
        dW /= num_train
        dW += 2 * reg * self.W

        return loss, dW


if __name__ == '__main__':
    dataset = datasets.load_breast_cancer()
    data = dataset.data
    target = dataset.target
    np.random.seed(int(time.time()))
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    X_train = data[:300]
    y_train = target[:300]
    X_test = data[300:]
    y_test = target[300:]

    svm = LinearSVM()
    svm.fit(X_train, y_train, num_iter=500, verbose=True)
    print("Accuracy on testing set:", np.mean(y_test == svm.predict(X_test)))

    svc = SVC(gamma="auto", kernel="linear")
    svc.fit(X_train, y_train)
    print("Accuracy using sklearn on testing set:", np.mean(y_test == svc.predict(X_test)))
