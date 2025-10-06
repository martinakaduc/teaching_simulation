import numpy as np
from typing import List
from env import Hypothesis, Point


def softmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - a.max()
    exp = np.exp(a)
    return exp / exp.sum()


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


def set_random_seed(seed: int):
    np.random.seed(seed)


def generate_hypotheses(
    n_hypotheses: int, n_cluster: int, n_features: int
) -> List[Hypothesis]:
    hypotheses = []
    for _ in range(n_hypotheses):
        centroids = [
            Point(np.random.uniform(-n_cluster, n_cluster, size=n_features))
            for _ in range(n_cluster)
        ]
        radiuses = [
            np.random.uniform(0.5 * n_cluster, float(n_cluster))
            for _ in range(n_cluster)
        ]
        hypotheses.append(Hypothesis(centroids=centroids, radiuses=radiuses))
    return hypotheses
