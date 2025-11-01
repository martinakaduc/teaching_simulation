# TeachSim: Teaching Simulation Framework

A computational framework for simulating interactive teaching scenarios between a teacher and a student in a clustering hypothesis learning task. This project implements rational and naive agents with different teaching and learning strategies to study effective pedagogical approaches.

## Overview

TeachSim models the interaction between a teacher and a student where:
- The **teacher** knows the true hypothesis (clustering structure) and selects data points to show the student
- The **student** maintains beliefs over possible hypotheses and can query for additional data points
- Both agents model each other's reasoning processes (theory of mind)

The framework supports both **naive** and **rational** agents:
- **Naive agents**: Update beliefs using Bayesian inference only
- **Rational agents**: Use recursive reasoning to model the other agent's intentions and beliefs

## Features

- 🧠 **Agent Modeling**: Teacher and student agents with configurable reasoning modes
- 🎯 **Multiple Strategies**: Random, hypothesis-based, and uncertainty-based selection strategies
- 📊 **Clustering Environment**: Gaussian mixture or uniform with configurable dimensionality
- 📈 **Comprehensive Logging**: Integration with Weights & Biases for experiment tracking
- 🔬 **Multiple Experiments**: Pre-configured experiments comparing different agent combinations
- 📉 **Visualization Tools**: Statistical plotting for analyzing learning curves and convergence

## Installation

### Requirements

- Python 3.10+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/martinakaduc/teaching_simulation.git
cd teaching_simulation
```

2. Install dependencies:
```bash
cd teaching_simulation
pip install -r requirements.txt
```

### Dependencies

```
jsonargparse==4.41.0
matplotlib==3.10.0
numpy==1.26.4
PyYAML==6.0.3
tqdm==4.67.1
tueplots==0.2.1
wandb==0.22.2
```

## Project Structure

```
teaching_simulation/
├── agents.py                # Teacher and student agent implementations
├── env.py                   # Clustering environment and hypothesis definitions
├── main.py                  # Main simulation entry point
├── utils.py                 # Utility functions (entropy, softmax, etc.)
├── plot_stats.py            # Visualization and statistical analysis
├── plot_traces.py           # Step-by-step teaching process visualization
├── requirements.txt         # Python dependencies
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # This file
├── configs/                 # YAML configuration files for experiments
│   ├── exp1.*.yaml          # Teacher strategy experiments
│   ├── exp2.*.yaml          # Student mode experiments
│   ├── exp3.*.yaml          # Student strategy experiments
│   ├── exp4.*.yaml          # Teacher assumption experiments
│   ├── exp5.*.yaml          # Alpha/Beta parameter experiments
│   ├── exp6.*.yaml          # Belief count experiments
│   ├── exp7.*.yaml          # Interaction mode experiments
│   └── wnb_config.yaml      # Weights & Biases configuration
├── results/                 # Output directory for simulation results
├── traces/                  # Output directory for trace visualizations
└── wandb/                   # Weights & Biases logging data
```

## Usage

### Basic Simulation

Run a single simulation with default parameters:

```bash
python main.py --seed 42 --n_rounds 10
```

### Using Configuration Files

Run simulations with predefined experiment configurations:

```bash
python main.py --config configs/exp1.7.yaml
```

### Custom Parameters

Configure simulation parameters via command-line arguments:

```bash
python main.py \
  --seed 42 \
  --n_hypotheses 20 \
  --n_clusters 2 \
  --n_features 2 \
  --n_samples 1000 \
  --n_rounds 100 \
  --data_initialization uniform \
  --interaction_mode active_interaction \
  --teacher_strategy hypothesis \
  --teacher_alpha 1.0 \
  --teacher_n_beliefs 100 \
  --teacher_student_mode_assumption rational \
  --teacher_student_strategy_assumption hypothesis \
  --student_mode rational \
  --student_strategy hypothesis \
  --student_beta 1.0 \
  --student_teacher_strategy_assumption hypothesis \
  --result_dir results
```

### Mock Test

Run a quick test with predefined simple hypotheses:

```bash
python main.py --mock_test --n_rounds 5
```

## Key Parameters

### Environment Parameters

- `n_hypotheses`: Number of possible hypotheses (default: 2)
- `n_clusters`: Number of clusters per hypothesis (default: 2)
- `n_features`: Dimensionality of data points (default: 2)
- `n_samples`: Number of data points to generate (default: 100)
- `data_initialization`: Data generation method
  - `uniform`: Uniform distribution initialization
  - `normal`: Normal/Gaussian distribution initialization (default)

### Teacher Parameters

- `teacher_strategy`: How teacher selects data points
  - `random`: Random selection
  - `hypothesis`: Information-theoretic selection based on expected belief updates
- `teacher_alpha`: Rationality parameter (higher = more rational)
- `teacher_n_beliefs`: Number of belief particles for sampling
- `teacher_student_mode_assumption`: Teacher's assumption about student (`naive` or `rational`)
- `teacher_student_strategy_assumption`: Teacher's assumption about student's querying strategy (`random`, `hypothesis`, or `uncertainty`)

### Student Parameters

- `student_mode`: Student reasoning mode
  - `naive`: Bayesian belief updates only
  - `rational`: Models teacher's pedagogical intentions
- `student_strategy`: How student queries new data points
  - `random`: Random queries
  - `hypothesis`: Hypothesis-driven queries
  - `uncertainty`: Uncertainty-reduction queries
- `student_beta`: Rationality parameter (higher = more rational)
- `student_teacher_strategy_assumption`: Student's assumption about teacher's strategy (`random` or `hypothesis`)

### Simulation Parameters

- `seed`: Random seed for reproducibility (default: 42)
- `n_rounds`: Number of interaction rounds (default: 100)
- `interaction_mode`: Mode of interaction between teacher and student
  - `active_interaction`: Both teacher and student actively participate
  - `lazy_student`: Student only observes teacher's demonstrations without querying
  - `lazy_teacher`: Teacher demonstrates initially but student drives learning through queries
- `result_dir`: Output directory for results (default: `results`)

## Experiments

The framework includes several pre-configured experiments in the `configs/` directory:

### Experiment Types

1. **Experiment 1**: Teacher strategy comparison
   - `exp1.1.yaml`: Random teacher strategy
   - `exp1.2_2.2_3.3_4.1_5.2_6.3_7.4.yaml`: Hypothesis-based teacher strategy (baseline)

2. **Experiment 2**: Student mode comparison
   - `exp2.1.yaml`: Naive student
   - Baseline config: Rational student

3. **Experiment 3**: Student strategy comparison
   - `exp3.1.yaml`: Random student queries
   - `exp3.2.yaml`: Uncertainty-based student queries
   - `exp3.4.yaml`: Lazy student (no queries)
   - Baseline config: Hypothesis-based student queries

4. **Experiment 4**: Teacher assumptions about student
   - `exp4.2.yaml`: Teacher assumes student uses uncertainty strategy
   - `exp4.3.yaml`: Teacher assumes student uses random strategy
   - `exp4.4.yaml`: Teacher assumes naive student
   - Baseline config: Teacher assumes rational student with hypothesis strategy

5. **Experiment 5**: Rationality parameter sensitivity (α, β)
   - `exp5.1.yaml`: Low rationality (α, β = 0.1)
   - `exp5.3.yaml`: High rationality (α, β = 10)
   - Baseline config: Medium rationality (α, β = 1.0)

6. **Experiment 6**: Belief particle count sensitivity (K)
   - `exp6.1.yaml`: K = 10 beliefs
   - `exp6.2.yaml`: K = 50 beliefs
   - Baseline config: K = 100 beliefs

7. **Experiment 7**: Interaction modes with lazy teacher
   - `exp7.1.yaml`: Lazy teacher with random student strategy
   - `exp7.2.yaml`: Lazy teacher with uncertainty student strategy
   - `exp7.3.yaml`: Lazy teacher with hypothesis student strategy
   - Baseline config: Active interaction mode

### Running Experiments with Weights & Biases

Run parameter sweeps using W&B:

```bash
wandb sweep configs/wnb_config.yaml
wandb agent <your-entity>/<project-name>/<sweep-id>
```

### Visualization

Generate plots comparing different experimental conditions:

```bash
python plot_stats.py \
  --exp 1 \
  --seeds 2 3 5 7 11 13 17 19 23 29 \
  --n_rounds 100 \
  --result_dir results
```

Available experiments: 1-7 (see [Experiments](#experiments) section for details)

You can also use the provided `script.sh` (in the root directory) which contains commands to:
- Run all experiments with W&B sweeps
- Generate all statistical plots
- Create trace visualizations for all experiments

This produces:
1. **Line plot**: Probability of true hypothesis over iterations
2. **Bar plot**: Mean iterations to reach rank #1 (with error bars)

### Trace Visualization

Generate step-by-step visualizations of the teaching process for a single simulation:

```bash
python plot_traces.py \
  --config configs/exp1.2_2.2_3.3_4.1_5.2_6.3_7.4.yaml \
  --seed 42 \
  --n_rounds 100
```

Optional parameters:
- `--max_rounds`: Limit the number of rounds to plot (useful for long simulations)
- `--plot_every`: Plot every N rounds to reduce the number of images (e.g., `--plot_every 5`)

This creates a `traces/` directory containing:
1. **Step-by-step trace images**: PNG files showing each round of the teaching process
   - Left panel: Data space with true hypothesis, teacher-shown points, and student-queried points
   - Right panel: Student's belief distribution over all hypotheses
2. **Learning curve plot**: Shows probability and rank of true hypothesis over time

Example output directory structure:
```
traces/
└── seed42_teach[hypothesis-1.0-100-rational-hypothesis]_stud[rational-hypothesis-1.0-hypothesis]/
    ├── trace_round_000.png
    ├── trace_round_001.png
    ├── trace_round_002.png
    ├── ...
    └── learning_curve.png
```

The trace visualizations help you:
- Understand how the student's beliefs evolve over time
- See which data points the teacher chooses to demonstrate
- Observe the student's active learning queries
- Verify that the student converges to the correct hypothesis

## Output

### Simulation Results

Results are saved as pickle files in the `results/` directory with the following structure:

```python
{
    "configs": {...},                          # Configuration parameters
    "hypotheses": [...],                       # All hypotheses
    "true_hypothesis_index": int,              # Index of ground truth
    "data": [...],                             # Generated data points
    "student_beliefs": [...],                  # Student belief history
    "student_actions": [...],                  # Student query history
    "student_true_hypothesis_probs": [...],    # Probability of true hypothesis
    "student_true_hypothesis_ranks": [...],    # Rank of true hypothesis
    "teacher_beliefs": [...],                  # Teacher belief history
    "teacher_actions": [...]                   # Teacher demonstration history
}
```

### Weights & Biases Logging

The simulation automatically logs metrics to W&B (configured in `configs/wnb_config.yaml`):
- `true_belief_prob`: Probability student assigns to true hypothesis
- `true_belief_rank`: Rank of true hypothesis in student's belief
- `round`: Current interaction round

To use W&B:
1. Sign up at [wandb.ai](https://wandb.ai)
2. Run `wandb login` to authenticate
3. Update the project name in `main.py` (`wandb.init(project="teachsim")`)
4. Configure sweep parameters in `configs/wnb_config.yaml` for hyperparameter tuning

## Core Components

### ClusteringEnv (`env.py`)

Defines the clustering environment with:
- **Point**: N-dimensional data points
- **Hypothesis**: Gaussian mixture model with centroids and radii
- **Data generation**: Uniform or normal initialization
- **Likelihood computation**: P(y|x, θ) for cluster assignments

### TeacherAgent (`agents.py`)

Features:
- Maintains beliefs about student's beliefs (second-order theory of mind)
- Selects informative data points using utility-based reasoning
- Updates beliefs based on student's queries
- Supports both naive and rational student modeling

### StudentAgent (`agents.py`)

Features:
- Maintains beliefs over hypotheses
- Updates beliefs via Bayesian inference
- Optionally models teacher's pedagogical reasoning (rational mode)
- Queries data points to reduce uncertainty or test hypotheses

## Algorithm Overview

### Teaching Loop

The simulation supports three interaction modes:

**Active Interaction** (default):
1. **Teacher selects** data point x and label y to demonstrate
2. **Student observes** (x, y) and updates hypothesis beliefs
3. **Student queries** new data point (or passes)
4. **Teacher observes** student's query and updates beliefs about student
5. Repeat for N rounds

**Lazy Student**:
1. **Teacher selects** data point x and label y to demonstrate
2. **Student observes** (x, y) and updates hypothesis beliefs
3. **No student queries** - passive learning only
4. Repeat for N rounds

**Lazy Teacher**:
1. **Teacher selects** initial data points to demonstrate
2. **Student observes** (x, y) and updates hypothesis beliefs
3. **Student queries** new data points actively
4. **Teacher observes** student's query and updates beliefs about student
5. **Teacher provides** data point the student queried
6. Repeat 2-5 for N-1 rounds

### Belief Updates

**Naive Student**:
```
P(θ | D_t, x, y) ∝ P(θ | D_t) × P(y | x, θ)
```

**Rational Student** (models teacher's pedagogy):
```
P(θ | D_t, x, y) ∝ P(θ | D_t) × P(x, y | θ)
where P(x, y | θ) = P_teacher(x | θ) × P(y | x, θ)
```

## Performance Metrics

The framework tracks:
- **Probability of true hypothesis**: How confident the student is in the correct answer
- **Rank of true hypothesis**: Position of true hypothesis in student's ranked beliefs
- **Iterations to convergence**: Rounds needed to reach 95% confidence or rank #1
- **Belief entropy**: Uncertainty in student's beliefs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{teachsim2025,
  author = {Duc Q. Nguyen},
  title = {TeachSim: A Framework for Teaching Simulation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/martinakaduc/teaching_simulation}
}
```

## License

This project is developed for CS6101 coursework at the National University of Singapore (NUS).  
Licensed under the MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

For detailed contribution guidelines, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the author directly.
Email: nqduc@u.nus.edu
