# Monte Carlo Yatzy Simulations

A high-performance Monte Carlo simulation suite for analyzing Yatzy game statistics and probability distributions. This project enables statistical exploration of score outcomes, category probabilities, and convergence behavior across millions of simulated games.

## Game Rules

This simulation implements the **Scandinavian Yatzy** ruleset, which differs from the more common US (Hasbro) version of Yahtzee. Key differences include:
- **Bonus**: 50 points for an upper section score of 63 or more (vs. 35 points).
- **Straights**: Small Straight is strictly `1-2-3-4-5` (15 pts) and Large Straight is `2-3-4-5-6` (20 pts).
- **Full House**: The score is the sum of all dice (e.g., 6,6,6,5,5 scores 28) instead of a fixed 25 points.
- **Yatzy Bonus**: There are no bonus points for subsequent Yatzys in the same game.
- **Maximum Score**: 374 points (theoretical maximum)

---

### AI Strategy

The simulation employs a deterministic, "naïve greedy" AI for decision-making.

- **Re-rolls**: The AI's re-roll strategy is to always keep the dice face that appears most frequently in the current roll. This decision is context-blind and does not account for which categories have already been filled.
- **Scoring**: After the final roll, the AI chooses the available category that yields the highest possible score. A fixed priority list is used only to break ties when multiple categories would give the same score.

This project provides a high-performance command-line tool for running Monte Carlo simulations of the game Yatzy. It is built for statistical analysis, allowing users to explore score distributions, category probabilities, and the law of large numbers with respect to game outcomes.

## Key Features

- **High Performance**: Utilizes `Numba` for JIT compilation of the core game loop, achieving millions of simulations per second.
- **Parallel Execution**: Employs a multi-threaded, constant-memory architecture to efficiently scale across multiple CPU cores.
- **Two Analysis Modes**:
  1.  **Distribution Analysis**: Generate detailed statistics from a single, large batch of simulations.
  2.  **Deviation Study**: Analyze how observed probabilities converge toward theoretical values as the number of simulations increases.
- **Reproducible AI**: A deterministic, priority-based AI makes decisions for re-rolls and category scoring, ensuring consistent simulation logic.
- **Detailed Outputs**: Generates CSV and JSON files for easy analysis in other tools like R, Python (with Pandas), or spreadsheet software.
- **Pre-computed Lookup Tables**: Generates 7,776 dice roll outcomes (6^5) at startup for O(1) category scoring and re-roll decisions.
- **Statistical Validation**: Uses exact combinatorial probabilities (7,776 total outcomes) to measure convergence in deviation studies.

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
- `--reps`: Number of repetitions per simulation step (default: 6)
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

## AI Strategy

The simulation employs a deterministic, context-blind AI for decision-making:

- **Re-roll Strategy**: The AI always keeps the dice face that appears most frequently in the current roll. This greedy approach does not account for which categories have already been filled, making it a "naïve" strategy but consistent and reproducible.
- **Scoring Strategy**: After the final roll, the AI chooses the available category that yields the highest possible score. A fixed priority list (Sixes → Aces → Yatzy → Four of a Kind → ... → Chance) is used only to break ties when multiple categories would give the same score.

This deterministic behavior ensures that simulations are reproducible and makes the AI's decision-making easy to analyze and understand.

## Project Structure

```
YatzySuite.py           # Main simulation engine
README.md               # This file
results/                # Output directory for simulation results
  ├── dist_*.csv       # Distribution analysis outputs
  ├── study_*.csv      # Deviation study outputs
  └── meta_*.json      # Metadata for runs
```

## How It Works

1. **Lookup Table Generation**: At startup, all 7,776 possible dice roll outcomes (6^5) are pre-computed to determine scores and re-roll decisions for all categories.
2. **Game Simulation**: For each simulated game:
   - Roll 5 dice three times, with re-rolls determined by the AI strategy
   - After each roll, compute statistics on which categories are achievable
   - Place the final score in the highest-scoring available category
3. **Parallel Batching**: Multiple CPU cores process independent simulation batches simultaneously, with constant memory usage.
4. **Statistical Analysis**: Results are aggregated and analyzed to compute distributions, probabilities, and convergence metrics.
