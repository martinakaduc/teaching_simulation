from typing import List
import argparse
from tqdm import tqdm
import numpy as np
from env import ClusteringEnv, Hypothesis, Point
from agents import TeacherAgent, StudentAgent


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
        radiuses = [np.random.uniform(0.5, float(n_cluster)) for _ in range(n_cluster)]
        hypotheses.append(Hypothesis(centroids=centroids, radiuses=radiuses))
    return hypotheses


def main(args):
    set_random_seed(args.seed)  # Set a random seed for reproducibility

    # Define hypotheses
    if args.mock_test:
        hypothesis1 = Hypothesis(
            centroids=[Point([0, 0]), Point([5, 5])], radiuses=[1.0, 1.0]
        )
        hypothesis2 = Hypothesis(
            centroids=[Point([0, 5]), Point([5, 0])], radiuses=[1.5, 1.5]
        )

        hypotheses = [hypothesis1, hypothesis2]
        true_hypothesis = hypothesis1
        true_hypothesis_index = hypotheses.index(true_hypothesis)
    else:
        hypotheses = generate_hypotheses(
            n_hypotheses=args.n_hypotheses,
            n_cluster=args.n_clusters,
            n_features=args.n_features,
        )
        true_hypothesis_index = int(np.random.randint(len(hypotheses)))
        true_hypothesis = hypotheses[true_hypothesis_index]

    # Initialize environment
    env = ClusteringEnv(
        n_features=args.n_features,
        n_samples=args.n_samples,
        data_initialization=args.data_initialization,
    )
    data = env.reset(true_hypothesis)

    # Initialize agents
    teacher = TeacherAgent(
        data=data,
        hypotheses=hypotheses,
        true_hypothesis=true_hypothesis,
        strategy=args.teacher_strategy,
        student_strategy=args.teacher_student_strategy_assumption,
        student_mode=args.teacher_student_mode_assumption,
        env=env,
        alpha=args.teacher_alpha,
        n_beliefs=args.teacher_n_beliefs,
    )

    student = StudentAgent(
        mode=args.student_mode,
        beta=args.student_beta,
        strategy=args.student_strategy,
        teacher_strategy=args.student_teacher_strategy_assumption,
        data=data,
        hypotheses=hypotheses,
        env=env,
    )
    if args.student_mode == "rational":
        student.set_teacher_belief(teacher.belief)
    print(
        "Student's belief of the true hypothesis:",
        student.belief.probs[true_hypothesis_index],
    )

    # Start simulation loop
    n_rounds = len(data)
    for round in tqdm(range(n_rounds), desc="Simulation Progress"):
        print("========== Round", round + 1, "==========")

        # Teacher selects a data point to show
        x_t, y_t = teacher.select_data_point()
        print(f"Teacher selected point: {x_t} with label {y_t}")
        print(f"Index of selected point: {data.index(x_t)}")
        print(
            len(teacher.unused_data_indices),
            "data points left for teacher to choose from.",
        )

        # Student updates beliefs based on the shown data point
        student.update_belief(x_t=x_t, y_t=y_t)
        print(
            "Student's belief of the true hypothesis:",
            student.belief.probs[true_hypothesis_index],
        )
        print(
            "Rank of true hypothesis in student's belief:",
            np.argsort(-student.belief.probs).tolist().index(true_hypothesis_index) + 1,
        )
        print(f"Student has not seen {len(student.unused_data_indices)} data points.")

        # Student makes an action (query a new data point or passive)
        a_t = student.make_action()
        if a_t:
            print(f"Student queried point: {a_t}")
            print(f"Index of queried point: {data.index(a_t)}")
        else:
            print("Student took no action.")

        # Teacher updates beliefs about the student's beliefs
        teacher.update_belief(a_t=a_t)

        print("-" * 50)

    print("========== Simulation End ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teaching Simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mock_test",
        action="store_true",
        help="Run a mock test with predefined parameters",
    )
    parser.add_argument(
        "--n_hypotheses", type=int, default=2, help="Number of hypotheses"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=2, help="Number of clusters per hypothesis"
    )
    parser.add_argument("--n_features", type=int, default=2, help="Number of features")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument(
        "--data_initialization",
        type=str,
        default="normal",
        choices=["uniform", "normal"],
        help="Data initialization method",
    )
    parser.add_argument(
        "--teacher_strategy",
        type=str,
        default="hypothesis",
        choices=["random", "hypothesis"],
        help="Teacher strategy to select data points",
    )
    parser.add_argument(
        "--teacher_student_strategy_assumption",
        type=str,
        default="uncertainty",
        choices=["random", "hypothesis", "uncertainty"],
        help="Teacher assumption about student strategy for querying data points",
    )
    parser.add_argument(
        "--teacher_student_mode_assumption",
        type=str,
        default="naive",
        choices=["rational", "naive"],
        help="Teacher assumption about student mode",
    )
    parser.add_argument(
        "--teacher_alpha", type=float, default=1.0, help="Teacher alpha parameter"
    )
    parser.add_argument(
        "--teacher_n_beliefs", type=int, default=100, help="Number of teacher beliefs"
    )
    parser.add_argument(
        "--student_mode",
        type=str,
        default="naive",
        choices=["rational", "naive"],
        help="Student mode",
    )
    parser.add_argument(
        "--student_strategy",
        type=str,
        default="uncertainty",
        choices=["random", "hypothesis", "uncertainty"],
        help="Student strategy for querying data points",
    )
    parser.add_argument(
        "--student_teacher_strategy_assumption",
        type=str,
        default="hypothesis",
        choices=["random", "hypothesis"],
        help="Student assumption about teacher strategy to select data points",
    )
    parser.add_argument(
        "--student_beta", type=float, default=1.0, help="Student beta parameter"
    )
    args = parser.parse_args()
    main(args)
