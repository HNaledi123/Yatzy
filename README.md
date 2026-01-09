# Monte Carlo Yatzy Simulator

High-performance Monte Carlo simulations for **Scandinavian Yatzy**, focused on score distributions, category probabilities, and convergence behavior at large sample sizes.

## Ruleset (Scandinavian Yatzy)

Differences vs. US Yahtzee:

* **Upper bonus**: 50 points at ≥63 (not 35)
* **Straights**:

  * Small: `1-2-3-4-5` → 15
  * Large: `2-3-4-5-6` → 20
* **Full House**: Sum of dice (not fixed 25)
* **Yatzy**: No multiple-Yatzy bonuses
* **Max score**: 374

## AI Model

Deterministic, intentionally naïve, and reproducible.

* **Re-rolls**: Always keep the most frequent face value in the roll (no category awareness).
* **Scoring**: After roll 3, select the available category with the highest score.
* **Ties**: Broken by a fixed priority order.

This AI is not optimal and is not meant to be.

## What This Tool Does

* Runs **millions of games per second** using Numba-JIT
* Uses **constant memory**, parallel CPU execution
* Produces **CSV/JSON outputs** for external analysis
* Uses **exact combinatorial probabilities** for validation

## Analysis Modes

### 1. Distribution Analysis

Single large run.

Outputs:

* Score distributions (with bonus/Yatzy splits)
* Per-category hit probabilities (rolls 1–3)
* Summary metadata (mean score, throughput)

### 2. Deviation Study

Multiple runs at increasing sample sizes.

Outputs:

* Absolute deviation from exact probabilities
* Per-category convergence behavior
* Timing statistics per step

## Prerequisites

Python 3.8+

```bash
pip install numpy numba
```

## Usage

### Distribution analysis

```bash
python YatzySuite.py --n 1000000
```

### Deviation study

```bash
python YatzySuite.py --study 1000,10000,100000 --reps 10
```

### Custom output directory

```bash
python YatzySuite.py --n 500000 --output my_results
```

## Output Files

```
results/
  dist_scores_*.csv
  dist_categories_*.csv
  study_deviation_*.csv
  study_summary_*.csv
  meta_*.json
```

## Implementation Notes (Brief)

* All 7,776 dice states are precomputed at startup
* Category scoring and re-roll decisions are O(1)
* Parallel execution uses bounded worker batches