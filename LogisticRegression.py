#! /usr/bin/env python3
import numpy as np
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html


class LogisticRegression:

    def __init__(self, theta=None, c=5000, alpha=0.005):
        self.theta = np.array(theta)
        self.c = c
        self.alpha = alpha

    @staticmethod
    def sigmoid(X):
        return np.array(1 / (1 + np.exp(-X)))

    def predict(self, x: np.array):
        try:
            one = np.ones(x.shape[0])
            xx = np.c_[np.ones(x.shape[0]), x]
            return self.sigmoid(xx.dot(self.theta))
        except Exception:
            print("predict error")
            print("x.shape: " + str(x.shape))
            print("theta.shape: " + str(self.theta.shape))
            print("X and thetas mismatch")
            return None

    
    def cost(self, x, y):
        eps = 1e-15
        y_hat = self.predict(x)
        y = np.squeeze(y)
        c = sum((y * np.log(y_hat + eps)) + ((1 - y) * np.log(1 - y_hat + eps))) / -len(y)
        return c

    def log_gradient(self, x, y):
        try:
            if (len(x) != y.shape[0]):
                print("x.shape: " + str(x.shape))
                print("y.shape: " + str(y.shape))
                print("X and Y size mismatch")
                return None
            if (x.T.shape[0] != len(self.theta) - 1):
                print("x.shape: " + str(x.shape))
                print("theta.shape: " + str(self.theta.shape))
                print("X and thetas mismatch")
                return None
            m = len(x)
            xc = np.c_[np.ones((len(x), 1)), x].T
            j = (1/m) * xc.dot(self.predict(x) - y)
            return j
            """ j = np.zeros(x.T.shape[0] + 1)
            for n in range(0, len(j)):
                if (n == 0):
                    j[n] = (1/m) * np.sum(self.predict(x) - y)
                else:
                    z = np.c_[np.ones(x.shape[0]), x]
                    j[n] = (np.sum((self.predict(x) - y).T.dot(z.T[n]).T)) / m
            return j """
        except Exception:
            return None
        
    def fit(self, x, y):
        m = len(x)
        X = np.c_[np.ones((len(x), 1)), x]
        y = np.squeeze(y)
        i = 0
        m1 = 1/m
        xt = np.transpose(X)
        while i < self.c:
            gradient = m1 * (np.dot(xt, (np.dot(X, self.theta) - y)))
            self.theta = self.theta - (self.alpha * gradient)
            i += 1
        return self.theta


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    mylr = LogisticRegression(np.array([2, 0.5, 7.1, -4.3, 2.09]))

    print(mylr.predict(X))
    print(mylr.cost(X,Y))
    mylr.fit(X,Y)
    print(mylr.theta)
    print(mylr.cost(X,Y))