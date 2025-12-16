# Monte Carlo Yatzy Simulations

This project provides a high-performance command-line tool for running Monte Carlo simulations of the game Yatzy. It is built for statistical analysis, allowing users to explore score distributions, category probabilities, and the law of large numbers with respect to game outcomes.

## Key Features

- **High Performance**: Utilizes `Numba` for JIT compilation of the core game loop, achieving millions of simulations per second.
- **Parallel Execution**: Employs a multi-threaded, constant-memory architecture to efficiently scale across multiple CPU cores.
- **Two Analysis Modes**:
  1.  **Distribution Analysis**: Generate detailed statistics from a single, large batch of simulations.
  2.  **Deviation Study**: Analyze how observed probabilities converge toward theoretical values as the number of simulations increases.
- **Reproducible AI**: A deterministic, priority-based AI makes decisions for re-rolls and category scoring, ensuring consistent simulation logic.
- **Detailed Outputs**: Generates CSV and JSON files for easy analysis in other tools like R, Python (with Pandas), or spreadsheet software.

## Installation

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install numpy numba
```

## Usage

This CLI tool provides high-performance Yatzy simulations for statistical analysis.

### Distribution Analysis

Analyze the distribution of scores and category probabilities over a large number of simulations.

```bash
python YatzySuite.py --n <number_of_simulations> [--output <output_directory>]
```

- `--n`: Number of Yatzy games to simulate
- `--output`: Output directory for results (default: "results")

**Outputs:**
- `dist_scores_<timestamp>.csv`: Score distribution with bonus and Yatzy flags
- `dist_categories_<timestamp>.csv`: Category hit probabilities across rolls
- `meta_dist_<timestamp>.json`: Metadata including mean score and performance

### Deviation Study

Study how observed category probabilities deviate from expected values across different simulation sizes.

```bash
python YatzySuite.py --study <simulation_steps> [--reps <repetitions>] [--output <output_directory>]
```

- `--study`: Comma-separated list of simulation counts (e.g., "1000,10000,100000")
- `--reps`: Number of repetitions per simulation step (default: 6). The first repetition of each step size is used as a JIT warmup and excluded from timing statistics. The `timing_stats_warmup_excluded` field in the metadata reports statistics on `--reps - 1` clean measurements.
- `--output`: Output directory for results (default: "results")

**Outputs:**
- `study_deviation_<timestamp>.csv`: Detailed deviation data for each category and repetition
- `study_summary_<timestamp>.csv`: Average deviations per category per simulation step
- `meta_study_<timestamp>.json`: Study metadata

### Examples

Run 1 million simulations for distribution analysis:
```bash
python YatzySuite.py --n 1000000
```

Run deviation study with 1K, 10K, and 100K simulations, 10 reps each:
```bash
python YatzySuite.py --study 1000,10000,100000 --reps 10
```

Save results to custom directory:
```bash
python YatzySuite.py --n 500000 --output my_results
```
