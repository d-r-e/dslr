import numpy as np
from collections import Counter

def count_(X):
    try:
        count = len(X[~np.isnan(X)])
        return count
    except Exception:
        return len(X)


def mean_(X):
    try:
        total = 0
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
        total += ((n - m) * (n - m))
    total = (total / len(X)) ** (1/2)
    return total


def min_(X):
    X = X[~np.isnan(X)]
    val = X[0]
    for n in X:
        if n < val:
            val = n
    return val


def max_(X):
    X = X[~np.isnan(X)]
    val = X[0]
    for n in X:
        if n > val:
            val = n
    return val


def percentile_(p, X):
    X = X[~np.isnan(X)]
    X.sort()
    length = (len(X) - 1) * p * 0.01
    left = np.floor(length)
    right = np.ceil(length)
    if left == right:
        return X[int(length)]
    a = X[int(left)] * (right - length)
    b = X[int(right)] * (length - left)
    return a + b

def range_(X):
    X = X[~np.isnan(X)]
    return (max(X) - min(X))

def mode_(X):
    n = len(X) 
    
    data = Counter(X) 
    get_mode = dict(data) 
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 
    return (max(mode))