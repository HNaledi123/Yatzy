# Monte Carlo Yatzy Simulations

## Installation

Ensure you have Python 3.x installed. Install required dependencies:

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
- `--reps`: Number of repetitions per simulation step (default: 5)
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
