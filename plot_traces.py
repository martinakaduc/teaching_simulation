import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from jsonargparse import ArgumentParser, ActionConfigFile
from tueplots import bundles, figsizes
from utils import create_result_path
from env import Point, Hypothesis
from tqdm import tqdm

plt.rcParams.update(bundles.iclr2024())


def plot_hypothesis(ax, hypothesis, alpha=0.3, marker="x", marker_size=8, label=None):
    """Plot a hypothesis as circles representing clusters."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (centroid, radius) in enumerate(
        zip(hypothesis.centroids, hypothesis.radiuses)
    ):
        color = color_cycle[i % len(color_cycle)]
        circle = Circle(
            centroid.coordinates,
            radius,
            color=color,
            alpha=alpha,
            fill=True,
            label=label if i == 0 else None,
        )
        ax.add_patch(circle)
        # Mark centroid
        ax.plot(
            centroid.coordinates[0],
            centroid.coordinates[1],
            marker=marker,
            color=color,
            markersize=marker_size,
            markeredgewidth=2,
        )


def plot_data_points(
    ax, data, shown_indices=None, shown_labels=None, queried_indices=None
):
    """Plot data points with different markers for shown and queried points."""
    data_array = np.array([p.coordinates for p in data])
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot all data points
    ax.scatter(
        data_array[:, 0],
        data_array[:, 1],
        c="lightgray",
        s=1,
        alpha=0.5,
        label="Unlabeled",
        zorder=1,
    )

    # Highlight shown points by teacher
    if shown_indices:
        shown_data = data_array[shown_indices]
        if shown_labels is not None:
            colors = [color_cycle[label % len(color_cycle)] for label in shown_labels]
        else:
            colors = "red"
        ax.scatter(
            shown_data[:-1, 0],
            shown_data[:-1, 1],
            c=colors[:-1],
            s=10,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="Teacher shown" if len(shown_data) > 1 else "",
            zorder=3,
        )
        ax.scatter(
            shown_data[-1:, 0],
            shown_data[-1:, 1],
            c=colors[-1:],
            s=50,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="Teacher shown" if len(shown_data) == 1 else "",
            zorder=3,
        )

    # Highlight queried points by student
    if queried_indices:
        queried_data = data_array[queried_indices]
        ax.scatter(
            queried_data[:-1, 0],
            queried_data[:-1, 1],
            c="red",
            s=10,
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            label="Student queried" if len(queried_data) > 1 else "",
            zorder=2,
        )
        ax.scatter(
            queried_data[-1:, 0],
            queried_data[-1:, 1],
            c="red",
            s=50,
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            label="Student queried" if len(queried_data) == 1 else "",
            zorder=2,
        )


def plot_belief_distribution(
    ax, beliefs, hypotheses, true_hypothesis_index, round_num, x_lim, y_lim
):
    """Plot histogram of belief distribution over hypotheses."""
    n_hypotheses = len(beliefs)
    half = (n_hypotheses + 1) // 2  # Ceiling division for odd number of hypotheses
    x_lim = (x_lim[0] / 1.2, x_lim[1] / 1.2)
    y_lim = (y_lim[0] / 1.2, y_lim[1] / 1.2)

    # Remove the main axis (we'll create a grid of subplots)
    ax.axis("off")
    ax.set_title(f"Student Belief Distribution (Round {round_num})")

    # Create a 4-row subplot within this axis
    subfig = ax.figure.add_subfigure(ax.get_subplotspec())

    # Create 4 subplots arranged vertically
    axes = subfig.subplots(
        4, 1, gridspec_kw={"height_ratios": [4, 1, 4, 1], "hspace": 0.0}
    )

    # First half hypotheses (Row 1)
    ax1 = axes[0]
    ax1.axis("off")
    fhypo_subfig = ax1.figure.add_subfigure(ax1.get_subplotspec())
    fhypo_axes = fhypo_subfig.subplots(1, half)

    for i in range(half):
        plot_hypothesis(
            fhypo_axes[i],
            hypotheses[i],
            alpha=0.2,
            marker="o",
            marker_size=2,
        )
        fhypo_axes[i].set_title(r"$\theta_{%d}$" % i, fontsize=6)
        fhypo_axes[i].set_xticks([])
        fhypo_axes[i].set_yticks([])
        fhypo_axes[i].set_xlim(x_lim)
        fhypo_axes[i].set_ylim(y_lim)
        fhypo_axes[i].set_aspect("equal")
        if i == true_hypothesis_index:
            for spine in fhypo_axes[i].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(2)

    # Beliefs for first half (Row 2)
    ax2 = axes[1]
    x_first = np.arange(half)
    colors_first = ["green" if i == true_hypothesis_index else "gray" for i in x_first]
    ax2.bar(
        x_first, beliefs[:half], color=colors_first, edgecolor="black", linewidth=0.5
    )
    ax2.set_xticks([])
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(labelsize=4)

    # Second half hypotheses (Row 3)
    ax3 = axes[2]
    ax3.axis("off")
    lhypo_subfig = ax3.figure.add_subfigure(ax3.get_subplotspec())
    lhypo_axes = lhypo_subfig.subplots(1, half)

    for i in range(half, n_hypotheses):
        plot_hypothesis(
            lhypo_axes[i - half],
            hypotheses[i],
            alpha=0.2,
            marker="o",
            marker_size=2,
        )
        lhypo_axes[i - half].set_title(r"$\theta_{%d}$" % i, fontsize=6)
        lhypo_axes[i - half].set_xticks([])
        lhypo_axes[i - half].set_yticks([])
        lhypo_axes[i - half].set_xlim(x_lim)
        lhypo_axes[i - half].set_ylim(y_lim)
        lhypo_axes[i - half].set_aspect("equal")
        if i == true_hypothesis_index:
            for spine in lhypo_axes[i - half].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(2)

    # Beliefs for second half (Row 4)
    ax4 = axes[3]
    x_second = np.arange(half, n_hypotheses)
    colors_second = [
        "green" if i == true_hypothesis_index else "gray" for i in x_second
    ]
    ax4.bar(
        x_second, beliefs[half:], color=colors_second, edgecolor="black", linewidth=0.5
    )
    ax4.set_xticks([])
    ax4.set_ylim(0, 1.0)
    ax4.tick_params(labelsize=4)


def plot_teaching_trace(
    result_file, output_dir, n_rounds=None, max_rounds=None, plot_every=1
):
    """
    Plot step-by-step visualization of the teaching process.

    Parameters
    ----------
    result_file : str
        Path to the pickle file containing simulation results.
    output_dir : str
        Directory to save the plots.
    max_rounds : int, optional
        Maximum number of rounds to plot. If None, plots all rounds.
    plot_every : int, optional
        Plot every N rounds to reduce number of plots (default: 1).
    """
    # Load results
    with open(result_file, "rb") as f:
        result_buffer = pickle.load(f)

    # Extract data
    data = [Point(p["coordinates"]) for p in result_buffer["data"]]
    hypotheses = [
        Hypothesis(
            centroids=[Point(c["coordinates"]) for c in h["centroids"]],
            radiuses=h["radiuses"],
        )
        for h in result_buffer["hypotheses"]
    ]
    true_hypothesis_index = result_buffer["true_hypothesis_index"]
    true_hypothesis = hypotheses[true_hypothesis_index]

    student_beliefs = result_buffer["student_beliefs"]
    teacher_actions = result_buffer["teacher_actions"]
    student_actions = result_buffer["student_actions"]

    if n_rounds is None:
        n_rounds = len(teacher_actions) - 1  # Exclude initial None action
    if max_rounds is not None:
        n_rounds = min(n_rounds, max_rounds)

    # Determine plot bounds
    data_array = np.array([p.coordinates for p in data])
    x_min, x_max = data_array[:, 0].min(), data_array[:, 0].max()
    y_min, y_max = data_array[:, 1].min(), data_array[:, 1].max()
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track shown and queried indices
    shown_indices = []
    shown_labels = []
    queried_indices = []

    for round_num in range(0, n_rounds + 1):
        teacher_action = teacher_actions[round_num]
        student_action = student_actions[round_num]

        # Update tracking lists
        if teacher_action is not None and teacher_action["x"] is not None:
            x_t = Point(teacher_action["x"]["coordinates"])
            if x_t in data:
                idx = data.index(x_t)
                if idx not in shown_indices:
                    shown_indices.append(idx)
                    shown_labels.append(teacher_action["y"])

        if student_action is not None and student_action["x"] is not None:
            a_t = Point(student_action["x"]["coordinates"])
            if a_t in data:
                idx = data.index(a_t)
                if idx not in queried_indices:
                    queried_indices.append(idx)

    # Create plots for each round
    for round_num in tqdm(range(0, n_rounds + 1, plot_every), desc="Plotting rounds"):
        if round_num == 0:
            # Initial state
            # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            fig = plt.figure()
            iclr_size = figsizes.iclr2024()["figure.figsize"]
            n_hypotheses_per_row = (len(hypotheses) + 1) // 2
            new_width = iclr_size[0] * (
                0.4 * 1.2 + 0.6 * 1.2 * n_hypotheses_per_row / 5
            )
            fig.set_size_inches(new_width, iclr_size[1] * 1.6)

            # Create 2 subplots arranged vertically, using height_ratios
            new_ratio = 0.4 / (0.4 + 0.6 * n_hypotheses_per_row / 5)
            ax1, ax2 = fig.subplots(
                1,
                2,
                gridspec_kw={
                    "width_ratios": [new_ratio, 1 - new_ratio]
                },  # not width_ratios since layout is vertical
            )

            # Left: Data space with true hypothesis
            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_aspect("equal")
            ax1.set_title("Initial State - True Hypothesis")

            plot_hypothesis(ax1, true_hypothesis, alpha=0.2)
            plot_data_points(ax1, data)
            ax1.legend(loc="upper right")

            # Right: Belief distribution
            initial_beliefs = np.array(student_beliefs[0]["probs"])
            plot_belief_distribution(
                ax2, initial_beliefs, hypotheses, true_hypothesis_index, 0, x_lim, y_lim
            )

            plt.savefig(
                os.path.join(output_dir, f"trace_round_{round_num:03d}.png"), dpi=300
            )
            plt.close()

        else:
            # Teaching interaction round
            student_belief = np.array(student_beliefs[round_num]["probs"])

            # Create figure
            fig = plt.figure()
            iclr_size = figsizes.iclr2024()["figure.figsize"]
            n_hypotheses_per_row = (len(hypotheses) + 1) // 2
            new_width = iclr_size[0] * (
                0.4 * 1.2 + 0.6 * 1.2 * n_hypotheses_per_row / 5
            )
            fig.set_size_inches(new_width, iclr_size[1] * 1.6)

            # Create 2 subplots arranged vertically, using height_ratios
            new_ratio = 0.4 / (0.4 + 0.6 * n_hypotheses_per_row / 5)
            ax1, ax2 = fig.subplots(
                1,
                2,
                gridspec_kw={
                    "width_ratios": [new_ratio, 1 - new_ratio]
                },  # not width_ratios since layout is vertical
            )
            # Left: Data space
            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_aspect("equal")
            ax1.set_title(f"Round {round_num} - Data Space")

            plot_hypothesis(ax1, true_hypothesis, alpha=0.2)
            plot_data_points(
                ax1,
                data,
                shown_indices=shown_indices[:round_num],
                shown_labels=shown_labels[:round_num],
                queried_indices=queried_indices[:round_num],
            )
            ax1.legend(loc="upper right")

            # Right: Belief distribution
            plot_belief_distribution(
                ax2,
                student_belief,
                hypotheses,
                true_hypothesis_index,
                round_num,
                x_lim,
                y_lim,
            )

            plt.savefig(
                os.path.join(output_dir, f"trace_round_{round_num:03d}.png"), dpi=300
            )
            plt.close()

    print(f"Created {(n_rounds // plot_every) + 1} trace plots in {output_dir}")


def plot_learning_curve(result_file, output_file):
    """
    Plot the learning curve showing probability of true hypothesis over time.

    Parameters
    ----------
    result_file : str
        Path to the pickle file containing simulation results.
    output_file : str
        Path to save the output plot.
    """
    with open(result_file, "rb") as f:
        result_buffer = pickle.load(f)

    true_probs = result_buffer["student_true_hypothesis_probs"]
    true_ranks = result_buffer["student_true_hypothesis_ranks"]
    n_hypotheses = len(result_buffer["hypotheses"])
    true_probs[0] = 1.0 / n_hypotheses  # Initial uniform belief
    true_ranks[0] = n_hypotheses  # Initial rank

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    iclr_size = figsizes.iclr2024()["figure.figsize"]
    fig.set_size_inches(iclr_size[0], iclr_size[1] * 1.5)

    # Plot probability
    ax1.plot(true_probs, marker="o", markevery=max(len(true_probs) // 10, 1))
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Probability")
    ax1.set_title("Probability of True Hypothesis")
    ax1.axhline(y=0.95, color="r", linestyle="--", linewidth=1)
    ax1.text(0.5, 0.96, r"95\% threshold", color="r")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot rank
    ax2.plot(
        true_ranks,
        marker="o",
        markevery=max(len(true_ranks) // 10, 1),
        color="green",
    )
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Rank")
    ax2.set_title("Rank of True Hypothesis")
    ax2.axhline(y=1, color="r", linestyle="--", linewidth=1)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Learning curve saved to {output_file}")


def main(args):
    # Create result file path
    result_file = create_result_path(args)

    if not os.path.exists(result_file):
        print(f"Error: Result file not found: {result_file}")
        return

    # Create output directory for traces
    output_dir = os.path.join(
        "traces",
        f"seed{args.seed}_teach[{args.teacher_strategy}-{args.teacher_alpha}-{args.teacher_n_beliefs}-"
        f"{args.teacher_student_mode_assumption}-{args.teacher_student_strategy_assumption}]_"
        f"stud[{args.student_mode}-{args.student_strategy}-{args.student_beta}-"
        f"{args.student_teacher_strategy_assumption}]",
    )

    # Plot step-by-step traces
    print("Generating step-by-step traces...")
    plot_teaching_trace(
        result_file,
        output_dir,
        n_rounds=args.n_rounds,
        max_rounds=args.max_rounds,
        plot_every=args.plot_every,
    )

    # Plot learning curve
    print("Generating learning curve...")
    learning_curve_file = os.path.join(output_dir, "learning_curve.png")
    plot_learning_curve(result_file, learning_curve_file)


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot Teaching Simulation Traces")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        default="uncertainty",
        choices=["random", "hypothesis", "uncertainty"],
        help="Teacher assumption about student strategy",
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
        default="uncertainty",
        choices=["random", "hypothesis", "uncertainty"],
        help="Student strategy for querying data points",
    )
    parser.add_argument(
        "--student_beta", type=float, default=1.0, help="Student beta parameter"
    )
    parser.add_argument(
        "--student_teacher_strategy_assumption",
        type=str,
        default="hypothesis",
        choices=["random", "hypothesis"],
        help="Student assumption about teacher strategy",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Output directory with simulation results",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=None,
        help="Maximum number of rounds to plot (default: all)",
    )
    parser.add_argument(
        "--plot_every",
        type=int,
        default=1,
        help="Plot every N rounds to reduce number of plots",
    )
    args = parser.parse_args()

    main(args)
