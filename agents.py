from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from env import Point, Hypothesis, ClusteringEnv
from utils import entropy


class TeacherBelief:
    def __init__(
        self, student_beliefs: List["StudentBelief"], probs: NDArray[np.float_]
    ):
        assert len(student_beliefs) == len(
            probs
        ), "Student beliefs and probabilities must be of same length"
        self.student_beliefs = student_beliefs
        self.probs = probs

    def to_dict(self):
        return {
            "student_beliefs": [sb.to_dict() for sb in self.student_beliefs],
            "probs": self.probs.tolist(),
        }

    def __str__(self) -> str:
        return (
            f"TeacherBelief(student_beliefs={self.student_beliefs}, probs={self.probs})"
        )


class StudentBelief:
    def __init__(self, hypotheses: List[Hypothesis], probs: NDArray[np.float_]):
        assert len(hypotheses) == len(
            probs
        ), "Hypotheses and probabilities must be of same length"
        self.hypotheses = hypotheses
        self.probs = probs

    def to_dict(self):
        return {
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "probs": self.probs.tolist(),
        }

    def __str__(self) -> str:
        return f"StudentBelief(hypotheses={self.hypotheses}, probs={self.probs})"


class TeacherAgent:
    def __init__(
        self,
        data: List[Point],
        hypotheses: List[Hypothesis],
        true_hypothesis: Hypothesis | None,
        strategy: str,
        student_strategy: str,
        student_mode: str,
        env: ClusteringEnv,
        alpha: float = 1.0,
        n_beliefs: int = 100,
    ):
        self.alpha = alpha
        self.data = data
        self.env = env
        self.hypotheses = hypotheses
        self.true_hypothesis = true_hypothesis
        assert strategy in ["random", "hypothesis"], "Invalid teacher strategy"
        self.strategy = strategy
        assert student_strategy in [
            "random",
            "hypothesis",
            "uncertainty",
        ], "Invalid student strategy"
        self.student_strategy = student_strategy
        assert student_mode in ["naive", "rational"], "Invalid student mode"
        self.student_mode = student_mode
        self.n_hypotheses = len(hypotheses)
        self.n_beliefs = n_beliefs
        self.n_clusters = len(hypotheses[0].centroids)
        self.unused_data_indices = list(range(len(data)))

        # Initialize teacher's belief about the student's belief
        # Sample `n_beliefs` teacher beliefs
        # Each teacher belief is a distribution over the hypotheses
        student_beliefs = []
        for _ in range(n_beliefs):
            belief_probs = np.random.dirichlet(np.ones(len(hypotheses)))
            student_beliefs.append(
                StudentBelief(hypotheses=hypotheses, probs=belief_probs)
            )
        self.belief = TeacherBelief(
            student_beliefs=student_beliefs, probs=np.ones(n_beliefs) / n_beliefs
        )

    def select_data_point(self) -> Tuple[Point, int]:
        """Select a data point to show to the student based on the current teacher belief."""
        assert (
            self.true_hypothesis is not None
        ), "True hypothesis must be set for teacher."
        assert (
            len(self.unused_data_indices) > 0
        ), "No unused data points left to select."

        if self.strategy == "random":
            chosen_idx = np.random.choice(self.unused_data_indices)
        elif self.strategy == "hypothesis":
            # Compute p(x | belief, theta*)
            p_x = self.p_x_given_belief_theta(
                theta_star=self.true_hypothesis,
                env=self.env,
                data=self.data,
                unused_data_indices=self.unused_data_indices,
                hypotheses=self.hypotheses,
                belief=self.belief,
                alpha=self.alpha,
            )
            chosen_idx = np.random.choice(list(p_x.keys()), p=list(p_x.values()))
        else:
            raise ValueError("Unknown teacher strategy")

        x_t = self.data[chosen_idx]
        y_t = self.env.sample_y_given_x_theta(x_t, self.true_hypothesis)

        # Update the student's belief after showing (x, y)
        TeacherAgent.update_student_beliefs(
            x_t=x_t,
            y_t=y_t,
            belief=self.belief,
            alpha=self.alpha,
            data=self.data,
            unused_data_indices=self.unused_data_indices,
            env=self.env,
            student_mode=self.student_mode,
        )
        self.unused_data_indices.remove(chosen_idx)
        return x_t, y_t

    def update_belief(self, a_t: Point | None) -> None:
        """Update the teacher's belief about the student's beliefs after observing student's action a_t."""
        TeacherAgent.update_belief_fn(
            a_t=a_t,
            belief=self.belief,
            beta=1.0,  # assuming beta=1.0 for teacher update
            data=self.data,
            unused_data_indices=self.unused_data_indices,
            env=self.env,
            student_strategy=self.student_strategy,
        )

    @classmethod
    def compute_utility(
        cls,
        x: Point,
        theta_star: Hypothesis,
        hypotheses: List[Hypothesis],
        env: ClusteringEnv,
        belief: TeacherBelief,
        eps: float = 1e-12,
    ) -> float:
        """
        Compute U(x; B_t, theta*) = E_{S_t ~ B_t}[ E_{y ~ p(·|x, theta*)}[ S_{t+1}(theta* | S_t, x, y) ] ].

        Parameters
        ----------
        x : Point
            The action or query point chosen by the teacher.
        theta_star : Hypothesis
            The true hypothesis.
        n_y_samples : int, default 512
            Number of Monte Carlo samples to approximate E_y.
        eps : float, default 1e-12
            Small constant to avoid division by zero during normalization.

        Returns
        -------
        float
            The expected posterior mass on theta_star, averaged over teacher beliefs
            and the observation model p(y | x, theta_star).
        """
        # Index of the true hypothesis in the hypothesis list
        try:
            true_idx = hypotheses.index(theta_star)
        except ValueError:
            raise ValueError("theta_star must be one of self.hypotheses.")

        # Precompute likelihoods for all hypotheses for each y sample
        # L has shape (n_clusters, n_hypotheses)
        n_hypotheses = len(hypotheses)
        n_clusters = len(theta_star.centroids)

        L = np.empty((n_clusters, n_hypotheses), dtype=float)
        for h_idx, h in enumerate(hypotheses):
            L[:, h_idx] = env.p_y_given_x_theta(x, h)

        # Outer expectation over teacher's belief distribution
        total_utility = 0.0
        for s_belief, w in zip(belief.student_beliefs, belief.probs):
            # Denominator for Bayes update for each y: sum_h prior[h] * L[i, h]
            denom = L @ s_belief.probs  # shape (n_clusters,)
            denom = np.maximum(denom, eps)  # numerical safety

            # Numerator for true hypothesis for each y: prior[true_idx] * L[i, true_idx]
            numer_true = (
                s_belief.probs[true_idx] * L[:, true_idx]
            )  # shape (n_clusters,)

            # Posterior mass on theta_star for each y, then average across samples
            post_true_per_y = numer_true / denom
            expected_post_true = float(post_true_per_y.mean())

            # Weight by teacher belief probability of this student belief
            total_utility += w * expected_post_true

        return float(total_utility)

    @classmethod
    def p_x_given_belief_theta(
        cls,
        theta_star: Hypothesis,
        env: ClusteringEnv,
        data: List[Point],
        unused_data_indices: List[int],
        hypotheses: List[Hypothesis],
        belief: TeacherBelief,
        alpha: float,
    ) -> Dict[int, float]:
        """
        Compute p(x | belief, theta*) ∝ exp(α * U(x; B_t, theta*))
        """
        utilities = {}
        for pidx in unused_data_indices:
            x = data[pidx]
            utilities[pidx] = np.exp(
                alpha
                * cls.compute_utility(
                    x=x,
                    theta_star=theta_star,
                    hypotheses=hypotheses,
                    env=env,
                    belief=belief,
                )
            )
        total = sum(utilities.values())
        return {pidx: util / total for pidx, util in utilities.items()}

    @classmethod
    def update_student_beliefs(
        cls,
        x_t: Point,
        y_t: int,
        belief: TeacherBelief,
        alpha: float,
        data: List[Point],
        unused_data_indices: List[int],
        env: ClusteringEnv,
        student_mode: str,
        eps: float = 1e-12,
    ) -> None:
        """
        Step 1:
        Update each student's belief about the true hypothesis after seeing (x_t, y_t):
            S_t(θ) ∝ S_{t-1}(θ) * p(y_t | x_t, θ)

        The teacher's belief over student beliefs remains unchanged.
        """
        for sbelief in belief.student_beliefs:
            StudentAgent.update_belief_fn(
                x_t=x_t,
                y_t=y_t,
                alpha=alpha,
                belief=sbelief,
                data=data,
                unused_data_indices=unused_data_indices,
                env=env,
                hypotheses=sbelief.hypotheses,
                mode=student_mode,
                teacher_belief=belief,
                eps=eps,
            )

    @classmethod
    def update_belief_fn(
        cls,
        a_t: Point | None,
        belief: TeacherBelief,
        beta: float,
        data: List[Point],
        unused_data_indices: List[int],
        env: ClusteringEnv,
        student_strategy: str,
        eps: float = 1e-12,
    ) -> None:
        """
        Step 2:
        Update the teacher's belief over student beliefs after observing student's action a_t:
            B_t(S_t) ∝ B_{t-1}(S_t) * p(a_t | S_t)

        where p(a_t | S_t) is modeled as the probability that a student with belief S_t
        would choose action a_t.
        """
        possible_actions = [data[pidx] for pidx in unused_data_indices] + [None]
        # Compute likelihood of action a_t under each student belief
        likelihoods = []
        for sbelief in belief.student_beliefs:
            p_a_given_S = StudentAgent.p_action_given_belief(
                all_actions=possible_actions,
                belief=sbelief,
                beta=beta,  # assuming beta=1.0 for student action model
                env=env,
                strategy=student_strategy,
            )
            P_a_given_S = p_a_given_S[a_t]

            likelihoods.append(P_a_given_S)

        prior = np.asarray(belief.probs, dtype=float)
        posterior = prior * np.array(likelihoods)
        posterior = np.maximum(posterior, eps)
        posterior /= posterior.sum()

        belief.probs = posterior  # update teacher belief over student beliefs


class StudentAgent:
    def __init__(
        self,
        mode: str,
        beta: float,
        strategy: str,
        data: List[Point],
        hypotheses: List[Hypothesis],
        env: ClusteringEnv,
        teacher_strategy: str,
    ):
        assert mode in ["naive", "rational"], "Invalid student mode"
        self.mode = mode
        assert strategy in [
            "random",
            "hypothesis",
            "uncertainty",
        ], "Invalid student strategy"
        assert len(hypotheses) > 0, "Hypotheses list cannot be empty"
        self.strategy = strategy
        assert teacher_strategy in [
            "random",
            "hypothesis",
        ], "Invalid teacher strategy assumption"
        self.teacher_strategy = teacher_strategy
        self.beta = beta
        self.data = data
        self.env = env
        self.unused_data_indices = list(range(len(data)))

        # Initialize belief over hypotheses
        # Student starts with a uniform prior over a predefined set of hypotheses
        self.hypotheses = hypotheses
        self.n_hypotheses = len(hypotheses)
        self.belief = StudentBelief(
            hypotheses=hypotheses, probs=np.ones(self.n_hypotheses) / self.n_hypotheses
        )
        if mode == "rational":
            self.teacher_model = TeacherAgent(
                data=data,
                hypotheses=hypotheses,
                true_hypothesis=None,
                strategy=teacher_strategy,
                student_strategy=strategy,
                student_mode=mode,
                env=env,
                alpha=1.0,
                n_beliefs=100,
            )
        else:
            self.teacher_model = None

    def set_teacher_belief(self, teacher_belief: TeacherBelief) -> None:
        """Set the teacher's belief about the student's beliefs."""
        if self.teacher_model is not None:
            self.teacher_model.belief = teacher_belief

    def update_belief(
        self,
        x_t: Point,
        y_t: int,
    ) -> None:
        """Update belief after observing (x_t, y_t)."""
        if self.teacher_model is not None:
            TeacherAgent.update_student_beliefs(
                x_t=x_t,
                y_t=y_t,
                belief=self.teacher_model.belief,
                alpha=self.teacher_model.alpha,
                data=self.data,
                unused_data_indices=self.unused_data_indices,
                env=self.env,
                student_mode=self.mode,
            )

        StudentAgent.update_belief_fn(
            x_t=x_t,
            y_t=y_t,
            belief=self.belief,
            hypotheses=self.hypotheses,
            mode=self.mode,
            alpha=self.teacher_model.alpha if self.teacher_model is not None else None,
            data=self.data,
            unused_data_indices=self.unused_data_indices,
            env=self.env,
            teacher_belief=(
                self.teacher_model.belief if self.teacher_model is not None else None
            ),
        )
        point_idx = self.data.index(x_t)
        if point_idx in self.unused_data_indices:
            self.unused_data_indices.remove(point_idx)
        else:
            raise ValueError("x_t must be in unused_data_indices")

    def make_action(self) -> Point | None:
        """Make an action (query a new data point or passive)."""
        possible_actions = [self.data[pidx] for pidx in self.unused_data_indices] + [
            None
        ]
        action_probs = StudentAgent.p_action_given_belief(
            all_actions=possible_actions,
            belief=self.belief,
            beta=self.beta,
            env=self.env,
            strategy=self.strategy,
        )
        actions, probs = zip(*action_probs.items())
        chosen_action = np.random.choice(actions, p=probs)

        # Update teacher's belief about student's beliefs
        if self.teacher_model is not None:
            TeacherAgent.update_belief_fn(
                a_t=chosen_action,
                belief=self.teacher_model.belief,
                beta=self.beta,
                data=self.data,
                unused_data_indices=self.unused_data_indices,
                env=self.env,
                student_strategy=self.strategy,
            )
        return chosen_action

    @classmethod
    def update_belief_fn(
        cls,
        x_t: Point,
        y_t: int,
        belief: StudentBelief,
        hypotheses: List[Hypothesis],
        mode: str,
        alpha: Optional[float],
        data: Optional[List[Point]],
        unused_data_indices: Optional[List[int]],
        env: Optional[ClusteringEnv],
        teacher_belief: Optional[TeacherBelief],
        eps: float = 1e-12,
    ) -> None:
        """
        Update belief after observing (x_t, y_t):
        If mode == "naive":
            S_t(θ) ∝ S_{t-1}(θ) * p(y_t | x_t, θ)
        If mode == "rational":
            S_t(θ) ∝ S_{t-1}(θ) * sum_{B_{t-1}} B_{t-1}(S_{t-1}) * p(x_t | B_{t-1}, θ) * p(y_t | x_t, θ)
        """
        prior = np.asarray(belief.probs, dtype=float)
        likelihoods = np.array(
            [ClusteringEnv.P_y_given_x_theta(y_t, x_t, h) for h in hypotheses]
        )  # (n_hypotheses,)
        posterior = prior * likelihoods

        if mode == "rational":
            assert (
                teacher_belief is not None
            ), "Teacher belief must be provided for rational mode"
            assert alpha is not None, "Alpha must be provided for rational mode"
            assert data is not None, "Data must be provided for rational mode"
            assert (
                unused_data_indices is not None
            ), "unused_data_indices must be provided for rational mode"
            assert env is not None, "Environment must be provided for rational mode"

            # Find index of x_t in data
            xidx = data.index(x_t)
            if xidx not in unused_data_indices:
                raise ValueError(
                    "x_t must be in unused_data_indices for rational update"
                )
            # Incorporate teacher's belief over student beliefs
            x_likelihoods = []
            for theta_star in hypotheses:
                P_x_given_belief = TeacherAgent.p_x_given_belief_theta(
                    theta_star=theta_star,
                    env=env,
                    data=data,
                    unused_data_indices=unused_data_indices,
                    hypotheses=hypotheses,
                    belief=teacher_belief,
                    alpha=alpha,
                )
                x_likelihoods.append(P_x_given_belief[xidx])
            x_likelihoods = np.array(x_likelihoods)  # (n_hypotheses,)
            posterior *= x_likelihoods

        posterior = np.maximum(posterior, eps)
        posterior /= posterior.sum()
        belief.probs = posterior  # update in place

    @classmethod
    def compute_utility(
        cls,
        a_t: Point | None,
        belief: StudentBelief,
        env: ClusteringEnv,
        strategy: str,
        eps: float = 1e-12,
    ) -> float:
        """
        Compute utility of each hypothesis under the student's belief and strategy.
        Here we use a simple heuristic:
        - For "random", all hypotheses have equal utility.
        - For "hypothesis", utility is proportional to belief in that hypothesis.
        - For "uncertainty", utility is inversely proportional to entropy of the belief.
        """

        if strategy == "random":
            return 1.0

        elif strategy == "hypothesis":
            # U(a; S_t) = max_{θ} E_{y ~ p(y | θ,a)}[p(θ | D_t ∪ {(a, y)})]
            if a_t is None:
                return max(belief.probs)

            # Precompute likelihoods for all hypotheses for each y sample
            # L has shape (n_clusters, n_hypotheses)
            L = []
            for hypothesis in belief.hypotheses:
                L.append(env.p_y_given_x_theta(a_t, hypothesis))
            L = np.array(L).T  # shape (n_clusters, n_hypotheses)

            n_hypotheses = len(belief.hypotheses)
            expected_posteriors = []
            for hidx, hypothesis, prob in zip(
                range(n_hypotheses), belief.hypotheses, belief.probs
            ):
                denom = L @ belief.probs  # (n_clusters,)
                denom = np.maximum(denom, eps)

                post_theta_per_y = prob * L[:, hidx] / denom  # (n_clusters,)
                expected_post_theta = float(post_theta_per_y.mean())
                expected_posteriors.append(expected_post_theta)
            return max(expected_posteriors)

        elif strategy == "uncertainty":
            # U(a;S_t)= -H_{θ} [ E_{y ~ p(y|θ,a)} [p(θ | D_t ∪ {(a,y)})] ]
            if a_t is None:
                return -entropy(belief.probs)

            # Precompute likelihoods for all hypotheses for each y sample
            # L has shape (n_clusters, n_hypotheses)
            L = []
            for hypothesis in belief.hypotheses:
                L.append(env.p_y_given_x_theta(a_t, hypothesis))
            L = np.array(L).T  # shape (n_clusters, n_hypotheses)

            n_hypotheses = len(belief.hypotheses)
            expected_posteriors = []
            for hidx, hypothesis, prob in zip(
                range(n_hypotheses), belief.hypotheses, belief.probs
            ):
                denom = L @ belief.probs  # (n_clusters,)
                denom = np.maximum(denom, eps)

                post_theta_per_y = prob * L[:, hidx] / denom  # (n_clusters,)
                expected_post_theta = float(post_theta_per_y.mean())
                expected_posteriors.append(expected_post_theta)
            expected_posteriors = np.array(expected_posteriors)
            expected_posteriors = np.maximum(expected_posteriors, eps)
            expected_posteriors /= expected_posteriors.sum()
            return -entropy(expected_posteriors)

        else:
            raise ValueError("Unknown strategy")

    @classmethod
    def p_action_given_belief(
        cls,
        all_actions: List[Point | None],
        belief: StudentBelief,
        beta: float,
        strategy: str,
        env: ClusteringEnv,
        eps: float = 1e-12,
    ) -> Dict[Point | None, float]:
        """
        Compute p(a_t | belief) ∝ exp(α * U(a_t; belief))
        where U(a_t; belief) is the utility of action a_t under the student's belief.
        """
        utilities = {}
        for action in all_actions:
            utilities[action] = np.exp(
                beta
                * cls.compute_utility(
                    a_t=action, belief=belief, env=env, strategy=strategy, eps=eps
                )
            )
        total = sum(utilities.values())
        return {action: util / total for action, util in utilities.items()}
