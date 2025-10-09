import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles, figsizes
from argparse import ArgumentParser
from utils import create_result_path

plt.rcParams.update(bundles.iclr2024())

EXP_CONFIGS = {
    1: {
        "TS_random w/ Naive Student": "exp1.1",
        "TS_hypothesis w/ Naive Student": "exp1.2",
        # "TS_hypothesis + SS_uncertainty": "exp1.3_2.1",
    },
    2: {
        "Naive Student": "exp1.3_2.1",
        "Rational Student": "exp2.2_3.3_4.1_5.1",
    },
    3: {
        "SS_random": "exp3.1",
        "SS_hypothesis": "exp3.2",
        "SS_uncertainty": "exp2.2_3.3_4.1_5.1",
    },
    4: {
        "TA_naive": "exp4.2",
        "TA_random": "exp4.3",
        "TA_hypothesis": "exp4.4",
        "TA_uncertainty": "exp2.2_3.3_4.1_5.1",
    },
    5: {
        r"$\alpha, \beta = 0.1$": "exp5.2",
        r"$\alpha, \beta = 1$": "exp2.2_3.3_4.1_5.1",
        r"$\alpha, \beta = 10$": "exp5.3",
    },
}


class ExpConfigurations:
    seed: int
    n_hypotheses: int
    n_clusters: int
    n_features: int
    n_samples: int
    data_initialization: str
    teacher_strategy: str
    teacher_n_beliefs: int
    teacher_alpha: float
    teacher_student_mode_assumption: str
    teacher_student_strategy_assumption: str
    student_beta: float
    student_mode: str
    student_strategy: str
    student_teacher_strategy_assumption: str = "hypothesis"
    result_dir: str


def load_config(config_file):
    config_path = os.path.join("configs", f"{config_file}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    exp_config = ExpConfigurations()
    for key, value in config.items():
        setattr(exp_config, key, value)
    exp_config.result_dir = args.result_dir
    return exp_config


def main(args):
    result_probs = {}
    result_ranks = {}
    for exp_name, config_file in EXP_CONFIGS[args.exp].items():
        if args.env == "medium":
            config_file += "_m"
        elif args.env == "difficult":
            config_file += "_d"
        print(f"Processing {exp_name} from {config_file}")
        exp_config = load_config(config_file)
        for seed in args.seeds:
            exp_config.seed = seed
            result_file = create_result_path(exp_config)
            with open(result_file, "rb") as f:
                result_buffer = pickle.load(f)

            # Extract true hypothesis probabilities
            student_true_hypothesis_probs = np.array(
                result_buffer["student_true_hypothesis_probs"]
            )  # (n_steps,)
            student_true_hypothesis_probs[0] = 1 / len(result_buffer["hypotheses"])

            if exp_name not in result_probs:
                result_probs[exp_name] = []
            result_probs[exp_name].append(student_true_hypothesis_probs)

            # Calculate iterations to reach rank #1
            iterations_to_rank1 = np.where(
                np.array(result_buffer["student_true_hypothesis_ranks"]) == 1
            )[0]
            if len(iterations_to_rank1) == 0:
                iterations_to_rank1 = len(student_true_hypothesis_probs)
            else:
                iterations_to_rank1 = (
                    iterations_to_rank1[0] + 1
                )  # +1 to convert index to iteration count
            if exp_name not in result_ranks:
                result_ranks[exp_name] = []
            result_ranks[exp_name].append(iterations_to_rank1)

    # Draw 1 plot containing 2 subplots
    # Line plot: Probability of true hypothesis over time
    # Column plot: Iteration to achieve rank #1 (with mean and std)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    iclr_size = figsizes.iclr2024()["figure.figsize"]
    fig.set_size_inches(iclr_size[0] * 1.2, iclr_size[1] * 1.2)

    # Line plot
    for exp_name, probs in result_probs.items():
        probs = np.array(probs, dtype=float)  # (n_seeds, n_steps)
        mean_probs = np.mean(probs, axis=0)
        std_probs = np.std(probs, axis=0)
        # Plot mean with markers every 10 steps
        ax1.plot(mean_probs, label=exp_name, marker="o", markevery=10)
        ax1.fill_between(
            range(len(mean_probs)),
            mean_probs - std_probs,
            mean_probs + std_probs,
            alpha=0.2,
        )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Probability")
    ax1.set_title("Probability of True Hypothesis in Student Belief")
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.95, color="r", linestyle="--")
    ax1.text(10.5, 0.96, r"95\% Probability", color="r")

    # Column plot
    exp_names = list(result_ranks.keys())
    iterations = [result_ranks[exp_name] for exp_name in exp_names]
    mean_iterations = [np.mean(iters) for iters in iterations]
    std_iterations = [np.std(iters) for iters in iterations]
    ax2.bar(exp_names, mean_iterations, yerr=std_iterations, capsize=5)
    ax2.set_ylabel(r"Iteration")
    ax2.set_title(r"True Hypothesis Reaches Rank \#1 in Student Belief")

    plt.savefig(
        os.path.join(args.result_dir, f"exp_{args.exp}_{args.env}.png"), dpi=300
    )
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Teaching Simulation")
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3, 4, 5], help="Experiment type"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["easy", "medium", "difficult"],
        default="easy",
        help="Environment type",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        help="Random seeds",
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
