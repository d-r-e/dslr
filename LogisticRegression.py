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
    

if __name__ == "__main__":
    lr = LogisticRegression([1,1])
    x = np.array([-4,2,0])
    print(lr.predict(x))
