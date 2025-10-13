from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from env import Point, Hypothesis, ClusteringEnv
from utils import entropy


class TeacherBelief:
    def __init__(
        self, student_beliefs: List["StudentBelief"], probs: NDArray[np.float64]
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
    def __init__(self, hypotheses: List[Hypothesis], probs: NDArray[np.float64]):
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
        data_likelihoods: NDArray[np.float64],
        alpha: float = 1.0,
        n_beliefs: int = 100,
    ):
        self.alpha = alpha
        self.env = env
        self.data = data
        self.data_likelihoods = data_likelihoods
        self.hypotheses = hypotheses
        self.true_hypothesis = true_hypothesis
        self.true_hypothesis_index = (
            hypotheses.index(true_hypothesis) if true_hypothesis is not None else -1
        )
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
        if self.student_mode == "rational":
            # Cache p(x | belief, theta) for all x and theta to avoid recomputation and infinite loops
            self.p_x_given_belief_theta_cache: NDArray[np.float64] = np.ones(
                (self.n_hypotheses, len(data))
            ) / len(
                data
            )  # shape (n_hypotheses, n_data)
        else:
            self.p_x_given_belief_theta_cache: NDArray[np.float64] = np.ones(
                (self.n_hypotheses, len(data))
            )

    def select_data_point(self) -> Tuple[Point, int]:
        """Select a data point to show to the student based on the current teacher belief."""
        assert (
            self.true_hypothesis_index > -1 and self.true_hypothesis is not None
        ), "True hypothesis must be set for teacher."
        assert (
            len(self.unused_data_indices) > 0
        ), "No unused data points left to select."

        if self.strategy == "random":
            chosen_idx = np.random.choice(self.unused_data_indices)

        elif self.strategy == "hypothesis":
            # Precompute likelihoods for all hypotheses for each y sample
            likelihoods = self.data_likelihoods[
                self.unused_data_indices
            ]  # shape (n_unused_data, n_clusters, n_hypotheses)

            if self.student_mode == "rational":
                p_allx_on_alltheta = []
                chosen_idx = None
                for theta_idx in range(self.n_hypotheses):
                    # Compute p(x | belief, theta)
                    p_x = self.p_x_given_belief_theta(
                        theta_star_idx=theta_idx,
                        unused_data_indices=self.unused_data_indices,
                        belief=self.belief,
                        strategy=self.strategy,
                        alpha=self.alpha,
                        likelihoods=likelihoods,
                        p_x_given_belief_theta_cache=self.p_x_given_belief_theta_cache,
                    )  # shape (n_unused_data,)
                    p_allx = np.full(len(self.data), np.nan)
                    p_allx[self.unused_data_indices] = p_x  # shape (n_data,)
                    p_allx_on_alltheta.append(p_allx)
                    if theta_idx == self.true_hypothesis_index:
                        chosen_idx = np.random.choice(self.unused_data_indices, p=p_x)

                self.p_x_given_belief_theta_cache = np.stack(
                    p_allx_on_alltheta
                )  # shape (n_hypotheses, n_data)

            elif self.student_mode == "naive":
                # Compute p(x | belief, theta*)
                p_x = self.p_x_given_belief_theta(
                    theta_star_idx=self.true_hypothesis_index,
                    unused_data_indices=self.unused_data_indices,
                    belief=self.belief,
                    strategy=self.strategy,
                    alpha=self.alpha,
                    likelihoods=likelihoods,
                    p_x_given_belief_theta_cache=self.p_x_given_belief_theta_cache,
                )  # shape (n_unused_data,)
                chosen_idx = np.random.choice(self.unused_data_indices, p=p_x)
            else:
                raise ValueError("Unknown student mode")
        else:
            raise ValueError("Unknown teacher strategy")

        assert chosen_idx is not None, "chosen_idx should be set"
        x_t = self.data[chosen_idx]
        y_t = self.env.sample_y_given_x_theta(x_t, self.true_hypothesis)

        # Update the student's belief after showing (x, y)
        TeacherAgent.update_student_beliefs(
            x_t=x_t,
            y_t=y_t,
            belief=self.belief,
            data=self.data,
            likelihoods=self.data_likelihoods,
            p_x_given_belief_theta_cache=self.p_x_given_belief_theta_cache,
            student_mode=self.student_mode,
            unused_data_indices=self.unused_data_indices,
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
            student_strategy=self.student_strategy,
            likelihoods=self.data_likelihoods[self.unused_data_indices],
        )

    @classmethod
    def compute_utility(
        cls,
        theta_star_idx: int,
        strategy: str,
        belief: TeacherBelief,
        likelihoods: NDArray[np.float64],
        prev_p_x_given_belief_theta: NDArray[np.float64],
        eps: float = 1e-12,
    ) -> float:
        """
        Compute U(x; B_t, theta*) = E_{S_t ~ B_t}[ E_{y ~ p(·|x, theta*)}[ S_{t+1}(theta* | S_t, x, y) ] ].

        Parameters
        ----------
        x : Point
            The action or query point chosen by the teacher.
        theta_star_idx : int
            The index of the true hypothesis in the hypotheses list.
        strategy : str
            The teacher's strategy ("random" or "hypothesis").
        belief : TeacherBelief
            The teacher's current belief over student beliefs.
        likelihoods : NDArray[np.float64]
            Precomputed likelihoods p(y | x, theta) for all y and theta.
            Shape (n_clusters, n_hypotheses).
        prev_p_x_given_belief_theta : NDArray[np.float64]
            Cached p(x | belief, theta) from previous computations, if available.
            Shape (n_hypotheses,).
        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        float
            The expected posterior mass on theta_star, averaged over teacher beliefs
            and the observation model p(y | x, theta_star).
        """
        if strategy == "random":
            return 1.0

        # Outer expectation over teacher's belief distribution
        total_utility = 0.0
        for s_belief, w in zip(belief.student_beliefs, belief.probs):
            # Denominator for Bayes update for each y: sum_h prior[h] * L[i, h]
            denom = likelihoods @ (
                s_belief.probs * prev_p_x_given_belief_theta
            )  # shape (n_clusters,)
            denom = np.maximum(denom, eps)  # numerical safety

            # Numerator for true hypothesis for each y: prior[true_idx] * L[i, true_idx]
            # Posterior mass on theta_star for each y, then average across samples
            post_true_per_y = (
                s_belief.probs[theta_star_idx]
                * prev_p_x_given_belief_theta[theta_star_idx]
                * likelihoods[:, theta_star_idx]
            ) / denom
            expected_post_true = np.sum(
                likelihoods[:, theta_star_idx] * np.log(post_true_per_y + eps)
            )

            # Weight by teacher belief probability of this student belief
            total_utility += w * expected_post_true

        return float(total_utility)

    @classmethod
    def p_x_given_belief_theta(
        cls,
        theta_star_idx: int,
        unused_data_indices: List[int],
        likelihoods: NDArray[np.float64],
        p_x_given_belief_theta_cache: NDArray[np.float64],
        belief: TeacherBelief,
        alpha: float,
        strategy: str,
        eps: float = 1e-12,
    ) -> NDArray[np.float64]:
        """
        Compute p(x | belief, theta*) ∝ exp(α * U(x; B_t, theta*))
        """

        utilities = []
        for uidx, pidx in enumerate(unused_data_indices):
            utilities.append(
                cls.compute_utility(
                    theta_star_idx=theta_star_idx,
                    strategy=strategy,
                    belief=belief,
                    likelihoods=likelihoods[uidx],
                    prev_p_x_given_belief_theta=p_x_given_belief_theta_cache[:, pidx],
                    eps=eps,
                )
            )
        U = np.array(utilities)
        exp_U = np.exp(alpha * U)
        return exp_U / np.sum(exp_U)

    @classmethod
    def update_student_beliefs(
        cls,
        x_t: Point,
        y_t: int,
        belief: TeacherBelief,
        data: List[Point],
        likelihoods: NDArray[np.float64],
        p_x_given_belief_theta_cache: NDArray[np.float64],
        student_mode: str,
        unused_data_indices: List[int],
    ) -> None:
        """
        Step 1:
        Update each student's belief about the true hypothesis after seeing (x_t, y_t):
            S_t(θ) ∝ S_{t-1}(θ) * p(y_t | x_t, θ)

        The teacher's belief over student beliefs remains unchanged.
        likelihoods: Precomputed likelihoods p(y | x, theta) for all y and theta.
            Shape (n_data, n_clusters, n_hypotheses).
        """
        for sbelief in belief.student_beliefs:
            StudentAgent.update_belief_fn(
                x_t=x_t,
                y_t=y_t,
                belief=sbelief,
                data=data,
                likelihoods=likelihoods,
                unused_data_indices=unused_data_indices,
                mode=student_mode,
                p_x_given_belief_theta_cache=p_x_given_belief_theta_cache,
            )

    @classmethod
    def update_belief_fn(
        cls,
        a_t: Point | None,
        belief: TeacherBelief,
        beta: float,
        data: List[Point],
        likelihoods: NDArray[np.float64],
        unused_data_indices: List[int],
        student_strategy: str,
        eps: float = 1e-12,
    ) -> None:
        """
        Step 2:
        Update the teacher's belief over student beliefs after observing student's action a_t:
            B_t(S_t) ∝ B_{t-1}(S_t) * p(a_t | S_t)

        where p(a_t | S_t) is modeled as the probability that a student with belief S_t
        would choose action a_t.

        likelihoods: Precomputed likelihoods p(y | x, theta) for all y and theta.
            Shape (n_unused_data, n_clusters, n_hypotheses).

        prev_p_x_given_belief_theta: Cached p(x | belief, theta) from previous computations, if available.
            Shape (n_hypotheses, n_data).
        """
        possible_actions = [data[pidx] for pidx in unused_data_indices] + [None]
        aidx = possible_actions.index(a_t)
        # Compute likelihood of action a_t under each student belief
        action_likelihoods = []
        for sbelief in belief.student_beliefs:
            p_a_given_S = StudentAgent.p_action_given_belief(
                belief=sbelief,
                beta=beta,
                likelihoods=likelihoods,
                strategy=student_strategy,
                unused_data_indices=unused_data_indices,
            )

            P_a_given_S = p_a_given_S[aidx]
            action_likelihoods.append(P_a_given_S)

        posterior = belief.probs * np.array(action_likelihoods)
        posterior = np.maximum(posterior, eps)
        posterior /= posterior.sum()
        belief.probs = posterior


class StudentAgent:
    def __init__(
        self,
        mode: str,
        beta: float,
        strategy: str,
        data: List[Point],
        hypotheses: List[Hypothesis],
        env: ClusteringEnv,
        data_likelihoods: NDArray[np.float64],
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
        self.env = env
        self.data = data
        self.data_likelihoods = data_likelihoods
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
                data_likelihoods=data_likelihoods,
                alpha=1.0,
                n_beliefs=1,  # Just a placeholder; The teacher belief will be set later!
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
            # Precompute likelihoods for all hypotheses for each y sample
            data_likelihoods = self.data_likelihoods[
                self.unused_data_indices
            ]  # shape (n_unused_data, n_clusters, n_hypotheses)

            p_allx_on_alltheta = []
            for theta_idx in range(len(self.hypotheses)):
                p_x = TeacherAgent.p_x_given_belief_theta(
                    theta_star_idx=theta_idx,
                    alpha=self.teacher_model.alpha,
                    belief=self.teacher_model.belief,
                    likelihoods=data_likelihoods,
                    p_x_given_belief_theta_cache=self.teacher_model.p_x_given_belief_theta_cache,
                    strategy=self.teacher_model.strategy,
                    unused_data_indices=self.unused_data_indices,
                )
                p_allx = np.full(len(self.data), np.nan)
                p_allx[self.unused_data_indices] = p_x  # shape (n_data,)
                p_allx_on_alltheta.append(p_allx)

            self.teacher_model.p_x_given_belief_theta_cache = np.stack(
                p_allx_on_alltheta
            )  # shape (n_hypotheses, n_data)

            TeacherAgent.update_student_beliefs(
                x_t=x_t,
                y_t=y_t,
                belief=self.teacher_model.belief,
                data=self.data,
                likelihoods=self.data_likelihoods,
                unused_data_indices=self.unused_data_indices,
                student_mode=self.mode,
                p_x_given_belief_theta_cache=self.teacher_model.p_x_given_belief_theta_cache,
            )

        StudentAgent.update_belief_fn(
            x_t=x_t,
            y_t=y_t,
            belief=self.belief,
            data=self.data,
            likelihoods=self.data_likelihoods,
            mode=self.mode,
            unused_data_indices=self.unused_data_indices,
            p_x_given_belief_theta_cache=(
                self.teacher_model.p_x_given_belief_theta_cache
                if self.teacher_model is not None
                else None
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

        if self.strategy == "random":
            action_idx = np.random.randint(len(possible_actions))
            chosen_action = possible_actions[action_idx]

        else:
            action_probs = StudentAgent.p_action_given_belief(
                belief=self.belief,
                beta=self.beta,
                likelihoods=self.data_likelihoods[
                    self.unused_data_indices
                ],  # shape (n_unused_data, n_clusters, n_hypotheses)
                unused_data_indices=self.unused_data_indices,
                strategy=self.strategy,
            )

            action_idx = np.random.choice(range(len(possible_actions)), p=action_probs)
            chosen_action = possible_actions[action_idx]

        # Update teacher's belief about student's beliefs
        if self.teacher_model is not None:
            TeacherAgent.update_belief_fn(
                a_t=chosen_action,
                belief=self.teacher_model.belief,
                beta=self.beta,
                data=self.data,
                unused_data_indices=self.unused_data_indices,
                likelihoods=self.data_likelihoods,
                student_strategy=self.strategy,
            )
        return chosen_action

    @classmethod
    def update_belief_fn(
        cls,
        x_t: Point,
        y_t: int,
        belief: StudentBelief,
        data: List[Point],
        likelihoods: NDArray[np.float64],
        mode: str,
        unused_data_indices: List[int],
        p_x_given_belief_theta_cache: NDArray[np.float64] | None,
    ) -> None:
        """
        Update belief after observing (x_t, y_t):
        If mode == "naive":
            S_t(θ) ∝ S_{t-1}(θ) * p(y_t | x_t, θ)
        If mode == "rational":
            S_t(θ) ∝ S_{t-1}(θ) * sum_{B_{t-1}} B_{t-1}(S_{t-1}) * p(x_t | B_{t-1}, θ) * p(y_t | x_t, θ)

        likelihoods: Precomputed likelihoods p(y | x, theta) for all y and theta.
            Shape (n_data, n_clusters, n_hypotheses).
        """
        # Find index of x_t in data
        xidx = data.index(x_t)
        if xidx not in unused_data_indices:
            raise ValueError("x_t must be in unused_data_indices for rational update")

        posterior = belief.probs * likelihoods[xidx, y_t, :]

        if mode == "rational":
            assert (
                p_x_given_belief_theta_cache is not None
            ), "p_x_given_belief_theta_cache must be provided for rational mode"

            x_likelihoods = p_x_given_belief_theta_cache[
                :, xidx
            ]  # shape (n_hypotheses,)
            posterior *= x_likelihoods

        posterior /= posterior.sum()
        belief.probs = posterior  # update in place

    @classmethod
    def compute_utility(
        cls,
        belief: StudentBelief,
        strategy: str,
        likelihoods: NDArray[np.float64] | None,
        eps: float = 1e-12,
    ) -> float:
        """
        Compute utility of each hypothesis under the student's belief and strategy.
        Here we use a simple heuristic:
        - For "random", all hypotheses have equal utility.
        - For "hypothesis", utility is proportional to belief in that hypothesis.
        - For "uncertainty", utility is inversely proportional to entropy of the belief.

        likelihoods: Shape (n_clusters, n_hypotheses) or None
        """

        if strategy == "random":
            return 1.0

        elif strategy == "hypothesis":
            # U(a; S_t) = max_{θ} E_{y ~ p(y | θ,a)}[p(θ | D_t ∪ {(a, y)})]
            if likelihoods is None:
                return max(belief.probs)

            n_hypotheses = len(belief.hypotheses)
            denom = likelihoods @ (belief.probs)  # (n_clusters,)
            denom = np.maximum(denom, eps)
            expected_posteriors = []
            for hidx, prob in zip(range(n_hypotheses), belief.probs):
                post_theta_per_y = prob * likelihoods[:, hidx] / denom  # (n_clusters,)
                expected_post_theta = float(
                    np.sum(likelihoods[:, hidx] * np.log(post_theta_per_y + eps))
                )
                expected_posteriors.append(expected_post_theta)
            return max(expected_posteriors)

        elif strategy == "uncertainty":
            # U(a;S_t)= -H_{θ} [ E_{y ~ p(y|θ,a)} [p(θ | D_t ∪ {(a,y)})] ]
            if likelihoods is None:
                return -entropy(belief.probs)

            n_hypotheses = len(belief.hypotheses)
            denom = likelihoods @ (belief.probs)  # (n_clusters,)
            denom = np.maximum(denom, eps)
            expected_posteriors = []
            for hidx, prob in zip(range(n_hypotheses), belief.probs):
                post_theta_per_y = prob * likelihoods[:, hidx] / denom  # (n_clusters,)
                expected_post_theta = float(
                    np.sum(likelihoods[:, hidx] * np.log(post_theta_per_y + eps))
                )
                expected_posteriors.append(expected_post_theta)
            expected_posteriors = np.exp(np.array(expected_posteriors))
            expected_posteriors /= expected_posteriors.sum()
            return -entropy(expected_posteriors)

        else:
            raise ValueError("Unknown strategy")

    @classmethod
    def p_action_given_belief(
        cls,
        belief: StudentBelief,
        beta: float,
        likelihoods: NDArray[np.float64],
        strategy: str,
        unused_data_indices: List[int],
        eps: float = 1e-12,
    ) -> NDArray[np.float64]:
        """
        Compute p(a_t | belief) ∝ exp(α * U(a_t; belief))
        where U(a_t; belief) is the utility of action a_t under the student's belief.

        likelihoods: Shape (n_unused_data, n_clusters, n_hypotheses)
        """
        utilities = []
        for uidx, pidx in enumerate(unused_data_indices + [None]):
            utilities.append(
                cls.compute_utility(
                    belief=belief,
                    likelihoods=likelihoods[uidx] if pidx is not None else None,
                    strategy=strategy,
                    eps=eps,
                )
            )
        U = np.array(utilities)
        exp_U = np.exp(beta * U)
        return exp_U / np.sum(exp_U)
