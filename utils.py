import numpy as np


def softmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - a.max()
    exp = np.exp(a)
    return exp / exp.sum()


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))
