import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray


class Point:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)

    def distance(self, other):
        return np.linalg.norm(self.coordinates - other.coordinates)

    def to_dict(self):
        return {"coordinates": self.coordinates.tolist()}

    def __str__(self) -> str:
        return f"Point({self.coordinates.tolist()})"


class Hypothesis:
    def __init__(self, centroids: List[Point], radiuses: List[float]):
        self.centroids = centroids
        self.radiuses = radiuses

    def to_dict(self):
        return {
            "centroids": [centroid.to_dict() for centroid in self.centroids],
            "radiuses": self.radiuses,
        }

    def __str__(self) -> str:
        return f"Hypothesis(centroids={self.centroids}, radiuses={self.radiuses})"


class ClusteringEnv:
    def __init__(
        self,
        n_features: int = 2,
        n_samples: int = 100,
        data_initialization: str = "normal",
    ):
        self.n_features: int = n_features
        self.n_samples: int = n_samples
        self.data: List[Point] = []
        self.true_hypothesis: Hypothesis | None = None
        self.data_initialization = data_initialization

    def reset(self, hypothesis: Hypothesis, seed: int = 42) -> List[Point]:
        assert hypothesis is not None, "Hypothesis must be provided for reset."
        rng = np.random.default_rng(seed)

        self.true_hypothesis = hypothesis
        self.data = self._generate_data(hypothesis, rng)

        return self.data

    def _generate_data(
        self, hypothesis: Hypothesis, rng: np.random.Generator
    ) -> List[Point]:
        data = []
        self.n_clusters = len(hypothesis.centroids)
        if self.data_initialization == "uniform":
            for _ in range(self.n_samples):
                point = rng.uniform(
                    -2 * self.n_clusters, 2 * self.n_clusters, size=self.n_features
                )
                data.append(Point(point))

        elif self.data_initialization == "normal":
            samples_per_cluster = self.n_samples // self.n_clusters
            for centroid, radius in zip(hypothesis.centroids, hypothesis.radiuses):
                for _ in range(samples_per_cluster):
                    point = rng.normal(
                        loc=centroid.coordinates, scale=radius, size=self.n_features
                    )
                    data.append(Point(point))

        else:
            raise ValueError(
                f"Unknown data initialization method: {self.data_initialization}"
            )

        return data

    @classmethod
    def p_y_given_x_theta(cls, x: Point, hypothesis: Hypothesis) -> NDArray[np.float_]:
        probabilities = []
        for centroid, radius in zip(hypothesis.centroids, hypothesis.radiuses):
            dist = x.distance(centroid)
            prob = np.exp(-(dist**2) / (2 * (radius**2)))
            probabilities.append(prob)

        probabilities = (
            np.array(probabilities)
            if sum(probabilities) > 0
            else np.ones(len(probabilities))
        )
        return probabilities / probabilities.sum()

    @classmethod
    def P_y_given_x_theta(cls, y: int, x: Point, hypothesis: Hypothesis) -> float:
        p_y_given_x = cls.p_y_given_x_theta(x, hypothesis)
        return p_y_given_x[y] if 0 <= y < len(p_y_given_x) else 0.0

    @classmethod
    def sample_y_given_x_theta(cls, x: Point, hypothesis: Hypothesis) -> int:
        probabilities = cls.p_y_given_x_theta(x, hypothesis)
        return np.random.choice(len(probabilities), p=probabilities)
