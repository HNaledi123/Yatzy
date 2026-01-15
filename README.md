# Monte Carlo Yatzy Simulator

High-performance Monte Carlo simulations for **Scandinavian Yatzy**, focused on score distributions, category probabilities, and convergence behavior at large sample sizes.

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
