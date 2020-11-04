#! /usr/bin/env python3
import numpy as np
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html


class LogisticRegression:

    def __init__(self, theta=None, c=100, lr=0.01):
        self.theta = np.array(theta)
        self._c = c
        self._lr = lr

    @staticmethod
    def sigmoid(X):
        return np.array(1 / (1 + np.exp(-X)))

    
    def predict(self, x: np.array):
        try:
            one = np.ones(x.size).reshape(-1,1)
            x = np.concatenate((one, x.reshape(-1,1)), axis=1)
            return self.sigmoid(x.dot(self.theta))
        except Exception:
            return None
    
    @staticmethod
    def cost(y, ypred, eps=1e-15):
        try:
            m = y.size
            lhs = y * (np.log(ypred))
            rhs = (1 - ypred)
            rhs = np.log(rhs)
            rhs = (1 - y) * np.log(1 - ypred)
            j = -1/m * (lhs + rhs)
            return float(j)
        except Exception:
            return None


if __name__ == "__main__":
    
    y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([[2], [0.5]])
    lr = LogisticRegression(theta1)
    y_hat1 = lr.predict(x1)
    cost = lr.cost(y1, y_hat1)
    print(cost)
