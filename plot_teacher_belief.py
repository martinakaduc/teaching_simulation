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


def plot_teacher_belief_distribution(
    fig,
    teacher_belief,
    hypotheses,
    true_hypothesis_index,
    round_num,
    x_lim,
    y_lim,
    top_k=3,
):
    """
    Plot the top K teacher beliefs (distributions over student beliefs).
    Each row shows one of the top K most probable student belief distributions.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to plot on.
    teacher_belief : dict
        Teacher belief with 'student_beliefs' and 'probs' keys.
    hypotheses : list
        List of all hypotheses.
    true_hypothesis_index : int
        Index of the true hypothesis.
    round_num : int
        Current round number.
    x_lim : tuple
        X-axis limits for hypothesis plots.
    y_lim : tuple
        Y-axis limits for hypothesis plots.
    top_k : int
        Number of top teacher beliefs to display (default: 3).
    """
    n_hypotheses = len(hypotheses)
    x_lim = (x_lim[0] / 1.2, x_lim[1] / 1.2)
    y_lim = (y_lim[0] / 1.2, y_lim[1] / 1.2)

    # Get top K teacher beliefs
    teacher_probs = np.array(teacher_belief["probs"])
    top_k_indices = np.argsort(teacher_probs)[-top_k:][
        ::-1
    ]  # Get top K, descending order

    # Remove the main axis (we'll create a grid of subplots)
    plt.axis("off")
    plt.title(f"Teacher Belief - Top {top_k} Student Beliefs (Round {round_num})")

    # Create subplots: top_k rows, each with 2 internal rows
    n_rows = top_k + 1
    height_ratios = [
        3.8,
    ] + [1.1] * top_k
    axes = fig.subplots(
        n_rows, 1, gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05}
    )

    # Row 1: Hypotheses
    ax1 = axes[0]
    ax1.axis("off")
    fhypo_subfig = ax1.figure.add_subfigure(ax1.get_subplotspec())
    fhypo_axes = fhypo_subfig.subplots(1, n_hypotheses)

    for i in range(n_hypotheses):
        plot_hypothesis(
            fhypo_axes[i],
            hypotheses[i],
            alpha=0.2,
            marker="o",
            marker_size=2,
        )
        fhypo_axes[i].set_title(r"$\theta_{%d}$" % i, fontsize=6, y=-0.5)
        fhypo_axes[i].set_xticks([])
        fhypo_axes[i].set_yticks([])
        fhypo_axes[i].set_xlim(x_lim)
        fhypo_axes[i].set_ylim(y_lim)
        fhypo_axes[i].set_aspect("equal")
        if i == true_hypothesis_index:
            for spine in fhypo_axes[i].spines.values():
                spine.set_edgecolor("green")
                spine.set_linewidth(2)

    for belief_idx, teacher_belief_idx in enumerate(top_k_indices):
        student_belief = np.array(
            teacher_belief["student_beliefs"][teacher_belief_idx]["probs"]
        )
        teacher_prob = teacher_probs[teacher_belief_idx]

        ax = axes[belief_idx + 1]
        fhypo_subfig = ax.figure.add_subfigure(ax.get_subplotspec())

        # Add belief rank and probability as a title
        if belief_idx == 0:
            fhypo_subfig.suptitle(
                f"Rank {belief_idx + 1}: $B(S) = {teacher_prob:.3f}$",
                fontsize=8,
                y=1.55,
                fontweight="bold",
            )
        else:
            fhypo_subfig.suptitle(
                f"Rank {belief_idx + 1}: $B(S) = {teacher_prob:.3f}$",
                fontsize=8,
                y=1.55,
            )

        # Teacher's Beliefs
        x_first = np.arange(n_hypotheses)
        colors_first = [
            "green" if i == true_hypothesis_index else "gray" for i in x_first
        ]
        ax.bar(
            x_first,
            student_belief,
            color=colors_first,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks([])
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=4)


def plot_teaching_trace_with_teacher_belief(
    result_file, output_dir, n_rounds=None, max_rounds=None, plot_every=1, top_k=3
):
    """
    Plot step-by-step visualization of the teaching process with teacher beliefs.

    Parameters
    ----------
    result_file : str
        Path to the pickle file containing simulation results.
    output_dir : str
        Directory to save the plots.
    n_rounds : int, optional
        Number of rounds to plot.
    max_rounds : int, optional
        Maximum number of rounds to plot. If None, plots all rounds.
    plot_every : int, optional
        Plot every N rounds to reduce number of plots (default: 1).
    top_k : int, optional
        Number of top teacher beliefs to display (default: 3).
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

    teacher_beliefs = result_buffer["teacher_beliefs"]
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
        teacher_belief = teacher_beliefs[round_num]

        # Create figure
        fig = plt.figure()
        iclr_size = figsizes.iclr2024()["figure.figsize"]
        n_hypotheses_per_row = (len(hypotheses) + 1) // 2
        # Adjust width to accommodate both data space and teacher belief visualization
        new_width = iclr_size[0] * (0.7 * 1.2 * n_hypotheses_per_row / 5)
        # Adjust height for top_k beliefs (each belief has 4 rows)
        new_height = iclr_size[1] * (
            0.64 + 0.16 * top_k
        )  # Scale based on number of beliefs
        fig.set_size_inches(new_width, new_height)

        # Teacher belief distribution (top K student beliefs)
        plot_teacher_belief_distribution(
            fig,
            teacher_belief,
            hypotheses,
            true_hypothesis_index,
            round_num,
            x_lim,
            y_lim,
            top_k=top_k,
        )

        plt.savefig(
            os.path.join(output_dir, f"teacher_belief_round_{round_num:03d}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"Created {(n_rounds // plot_every) + 1} teacher belief plots in {output_dir}"
    )


def main(args):
    # Create result file path
    result_file = create_result_path(args)

    if not os.path.exists(result_file):
        print(f"Error: Result file not found: {result_file}")
        return

    # Create output directory for traces
    output_dir = os.path.join(
        "traces",
        f"seed{args.seed}_"
        f"{'lazy' if args.interaction_mode == 'lazy_teacher' else ''}"
        f"teach[{args.teacher_strategy}-{args.teacher_alpha}-{args.teacher_n_beliefs}-"
        f"{args.teacher_student_mode_assumption}-{args.teacher_student_strategy_assumption}]_"
        f"{'lazy' if args.interaction_mode == 'lazy_student' else ''}"
        f"stud[{args.student_mode}-{args.student_strategy}-{args.student_beta}-"
        f"{args.student_teacher_strategy_assumption}]",
    )
    # output_dir = output_dir + "_teacher_belief"

    # Plot step-by-step traces with teacher beliefs
    print("Generating step-by-step teacher belief traces...")
    plot_teaching_trace_with_teacher_belief(
        result_file,
        output_dir,
        n_rounds=args.n_rounds,
        max_rounds=args.max_rounds,
        plot_every=args.plot_every,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot Teaching Simulation Teacher Beliefs")
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top teacher beliefs to display (default: 3)",
    )
    args = parser.parse_args()

    main(args)
