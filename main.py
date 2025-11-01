from jsonargparse import ArgumentParser, ActionConfigFile
from tqdm import tqdm
import os
import pickle
import numpy as np
import wandb
from env import ClusteringEnv, Hypothesis, Point
from agents import TeacherAgent, StudentAgent
from utils import create_result_path, set_random_seed, generate_hypotheses


def main(args):
    set_random_seed(args.seed)  # Set a random seed for reproducibility

    # Initialize buffer to store results
    result_buffer = {
        "configs": vars(args),
        "hypotheses": [],
        "true_hypothesis_index": -1,
        "data": [],
        "student_beliefs": [],
        "student_actions": [None],
        "student_true_hypothesis_probs": [None],
        "student_true_hypothesis_ranks": [None],
        "teacher_beliefs": [],
        "teacher_actions": [None],
    }
    result_file = create_result_path(args)

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
        hypotheses, true_hypothesis_index = generate_hypotheses(
            n_hypotheses=args.n_hypotheses,
            n_cluster=args.n_clusters,
            n_features=args.n_features,
            seed=42,
        )
        true_hypothesis = hypotheses[true_hypothesis_index]

    result_buffer["hypotheses"] = [h.to_dict() for h in hypotheses]
    result_buffer["true_hypothesis_index"] = true_hypothesis_index

    # Initialize environment
    env = ClusteringEnv(
        n_features=args.n_features,
        n_samples=args.n_samples,
        data_initialization=args.data_initialization,
    )
    data = env.reset(true_hypothesis, seed=42)
    data_likelihoods = env.compute_data_likelihoods(
        hypotheses
    )  # shape (n_samples, n_clusters + 1, n_hypotheses)
    result_buffer["data"] = [point.to_dict() for point in data]

    # Initialize agents
    teacher = TeacherAgent(
        data=data,
        hypotheses=hypotheses,
        true_hypothesis=true_hypothesis,
        strategy=args.teacher_strategy,
        student_strategy=args.teacher_student_strategy_assumption,
        student_mode=args.teacher_student_mode_assumption,
        interaction_mode=args.interaction_mode,
        env=env,
        data_likelihoods=data_likelihoods,
        alpha=args.teacher_alpha,
        n_beliefs=args.teacher_n_beliefs,
    )
    result_buffer["teacher_beliefs"].append(teacher.belief.to_dict())

    is_mismatched = (
        args.teacher_student_mode_assumption != args.student_mode
        or args.teacher_student_strategy_assumption != args.student_strategy
        or args.teacher_alpha != args.student_beta
    )
    student = StudentAgent(
        mode=args.student_mode,
        beta=args.student_beta,
        strategy=args.student_strategy,
        teacher_strategy=args.student_teacher_strategy_assumption,
        teacher_belief=teacher.belief,
        interaction_mode=args.interaction_mode,
        data=data,
        hypotheses=hypotheses,
        env=env,
        data_likelihoods=data_likelihoods,
        teacher_model=None if is_mismatched else teacher,
    )
    print(
        "Student's belief of the true hypothesis:",
        student.belief.probs[true_hypothesis_index],
    )
    result_buffer["student_beliefs"].append(student.belief.to_dict())

    # Start simulation loop
    for round in tqdm(range(args.n_rounds), desc="Simulation Progress"):
        print(f"\n{'='*20} ROUND {round + 1} {'='*20}\n")

        # Teacher selects a data point to show
        print("[Teacher] Selecting data point...")
        x_t, y_t = teacher.select_data_point()
        print(f"    → Selected Point #{data.index(x_t)}: {x_t} | Label: {y_t}\n")

        # Student updates beliefs based on the shown data point
        print("[Student] Updating belief...")
        student.update_belief(x_t=x_t, y_t=y_t)
        belief_true = student.belief.probs[true_hypothesis_index]
        rank_true = (
            np.argsort(-student.belief.probs).tolist().index(true_hypothesis_index) + 1
        )
        print(f"    → Belief in True Hypothesis: {belief_true:.4f}")
        print(f"    → Rank of True Hypothesis: {rank_true}\n")

        if args.interaction_mode in ["active_interaction", "lazy_teacher"]:
            # Student makes an action (query a new data point or passive)
            print("[Student] Making an action...")
            a_t = student.make_action()
            if a_t:
                print(f"    → Queried Point #{data.index(a_t)}: {a_t}\n")
            else:
                print("    → No action taken.\n")

            # Teacher updates beliefs about the student's beliefs
            print("[Teacher] Updating belief based on student's action...")
            teacher.update_belief(a_t=a_t)
            print(
                "    → Maximal teacher belief over student beliefs:",
                teacher.belief.probs.max(),
                "\n",
            )
        else:
            a_t = None
            print("[Interaction Mode] Lazy Student: No action taken.\n")

        # Store results of this round
        result_buffer["teacher_beliefs"].append(teacher.belief.to_dict())
        result_buffer["teacher_actions"].append({"x": x_t.to_dict(), "y": y_t})
        result_buffer["student_beliefs"].append(student.belief.to_dict())
        result_buffer["student_actions"].append({"x": a_t.to_dict() if a_t else None})
        result_buffer["student_true_hypothesis_probs"].append(belief_true)
        result_buffer["student_true_hypothesis_ranks"].append(rank_true)

        # Save intermediate results
        print("[SYSTEM] Saving intermediate results...\n")
        with open(result_file, "wb") as f:
            pickle.dump(result_buffer, f)

        wandb.log(
            {
                "true_belief_prob": belief_true,
                "true_belief_rank": rank_true,
                "round": round + 1,
            }
        )

        print("-" * 60)


if __name__ == "__main__":
    wandb.init(project="teachsim")

    parser = ArgumentParser(description="Teaching Simulation")
    parser.add_argument("--config", action=ActionConfigFile)
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
        "--n_rounds", type=int, default=100, help="Number of simulation rounds"
    )
    parser.add_argument(
        "--data_initialization",
        type=str,
        default="normal",
        choices=["uniform", "normal"],
        help="Data initialization method",
    )
    parser.add_argument(
        "--interaction_mode",
        type=str,
        default="lazy_student",
        choices=["active_interaction", "lazy_student", "lazy_teacher"],
        help="Interaction mode between teacher and student",
    )
    parser.add_argument(
        "--teacher_strategy",
        type=str,
        default="hypothesis",
        choices=["random", "hypothesis"],
        help="Teacher strategy to select data points",
    )
    parser.add_argument(
        "--teacher_alpha", type=float, default=1.0, help="Teacher alpha parameter"
    )
    parser.add_argument(
        "--teacher_n_beliefs", type=int, default=100, help="Number of teacher beliefs"
    )
    parser.add_argument(
        "--teacher_student_strategy_assumption",
        type=str,
        default="",
        choices=["", "random", "uncertainty", "hypothesis"],
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
        "--student_mode",
        type=str,
        default="naive",
        choices=["rational", "naive"],
        help="Student mode",
    )
    parser.add_argument(
        "--student_strategy",
        type=str,
        default="",
        choices=["", "random", "uncertainty", "hypothesis"],
        help="Student strategy for querying data points",
    )
    parser.add_argument(
        "--student_beta", type=float, default=1.0, help="Student beta parameter"
    )
    parser.add_argument(
        "--student_teacher_strategy_assumption",
        type=str,
        default="",
        choices=["", "random", "hypothesis"],
        help="Student assumption about teacher strategy to select data points",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Output directory to save simulation results",
    )
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    main(args)

    wandb.finish()
