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
            one = np.ones(x.shape[0])
            x = np.c_[np.ones(x.shape[0]),x]
            return self.sigmoid(x.dot(self.theta))
        except Exception:
            return None
    
    @staticmethod
    def loss(y: np.array, ypred: np.array, eps=1e-15):
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

    def log_gradient(self, x, y):
        if (x.shape[0] is not y.shape[0]):
            print("x.shape: "+ str(x.shape))
            print("y.shape: "+ str(y.shape))
            print("X and Y size mismatch")
            return None
        if (x.T.shape[0] != self.theta.shape[0] - 1):
            print("x.shape: "+ str(x.shape))
            print("theta.shape: "+ str(self.theta.shape))
            print("X and thetas mismatch")
            return None
        m = len(x)
        j = np.zeros(x.T.shape[0] + 1)
        for n in range(len(j)):
            if (n == 0):
                print(self.predict(x))
                j[n] = 1/m * np.sum(self.predict(x) - y.T)
        print("J:")
        print(j)
        print()
        return j
    
    


if __name__ == "__main__":
    x = np.array([4])
    lr = LogisticRegression(np.array([[2],[0.5]]))
    yprd = lr.predict(x)
    print(yprd)

    x2 = np.array([[4],[7.16], [3.2], [9.37],[0.56]])
    lr = LogisticRegression(np.array([[2],[0.5]]))
    ypred = lr.predict(x2)
    print(ypred)

    x3 = np.array([[0,2,3,4], [2,4,5,5], [1,3,2,7]])
    lr = LogisticRegression(np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]]))
    ypred = lr.predict(x3)
    print(ypred)
    """ y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([[2], [0.5]])
    lr = LogisticRegression(theta1)
    y_hat1 = lr.predict(x1)
    loss = lr.loss(y1, y_hat1)
    lr.log_gradient(x1, y1)

    y2 = np.array([[1],[0], [1], [0],[1]])
    x2 = np.array([[4],[7.16], [3.2], [9.37],[0.56]])
    lr = LogisticRegression(np.array([[2],[0.5]]))
    lr.log_gradient(x2, y2)

    y3 = np.array([[0],[1],[1]])
    x3 = np.array([[0,2,3,4], [2,4,5,5], [1,3,2,7]])
    lr = LogisticRegression(np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]]))
    lr.log_gradient(x3,y3) """