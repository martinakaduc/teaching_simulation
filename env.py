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
            n_partitions = int(self.n_samples ** (1 / self.n_features))
            remaining_points = self.n_samples - n_partitions**self.n_features
            ranges = np.linspace(
                [-2 * self.n_clusters] * self.n_features,
                [2 * self.n_clusters] * self.n_features,
                n_partitions + 1,
            ).T
            range_bounds = np.stack((ranges[:, :-1], ranges[:, 1:]), axis=-1)
            cartesian_idxs = np.array(
                np.meshgrid(*([list(range(n_partitions))] * self.n_features))
            ).T.reshape(-1, self.n_features)
            cartesian_rb = range_bounds[list(range(self.n_features)), cartesian_idxs]

            initial_points = np.concatenate(
                (
                    rng.uniform(
                        low=cartesian_rb[..., 0],
                        high=cartesian_rb[..., 1],
                        size=[n_partitions**self.n_features, self.n_features],
                    ),
                    rng.uniform(
                        low=-2 * self.n_clusters,
                        high=2 * self.n_clusters,
                        size=[remaining_points, self.n_features],
                    ),
                ),
                axis=0,
            )
            for point in initial_points:
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

    def compute_data_likelihoods(
        self, hypotheses: List[Hypothesis]
    ) -> NDArray[np.float64]:
        assert self.data, "Data must be initialized. Call reset() first."
        n_data = len(self.data)
        n_hypotheses = len(hypotheses)
        data_likelihoods = np.zeros((n_data, self.n_clusters, n_hypotheses))

        for h_idx, hypothesis in enumerate(hypotheses):
            for x_idx, x in enumerate(self.data):
                p_y_given_x = ClusteringEnv.p_y_given_x_theta(x, hypothesis)
                data_likelihoods[x_idx, :, h_idx] = p_y_given_x

        return data_likelihoods  # shape (n_data, n_clusters, n_hypotheses)

    @classmethod
    def p_y_given_x_theta(cls, x: Point, hypothesis: Hypothesis) -> NDArray[np.float64]:
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
        return p_y_given_x[y]

    @classmethod
    def sample_y_given_x_theta(cls, x: Point, hypothesis: Hypothesis) -> int:
        probabilities = cls.p_y_given_x_theta(x, hypothesis)
        return np.random.choice(len(probabilities), p=probabilities)
