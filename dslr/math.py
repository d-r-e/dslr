import numpy as np


def count_(X):
    try:
        count = len(X[~np.isnan(X)])
        return count
    except Exception:
        return len(X)


def mean_(X):
    total = 0
    try:
        X = X[~np.isnan(X)]
        for each in X:
            total += each
        return total / len(X)
    except Exception:
        return 0


def std_(X):
    X = X[~np.isnan(X)]
    m = mean_(X)
    total = 0
    for n in X:
        total = total + (n - m) ** 2
    return total


def min_(X):
    val = X[0]
    for n in X:
        if n < val:
            val = n
    return val
