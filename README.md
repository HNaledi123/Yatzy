# Monte Carlo Yatzy Simulator

High-performance Monte Carlo simulations for **Scandinavian Yatzy**, focused on score distributions, category probabilities, and convergence behavior at large sample sizes.

## Usage & Outputs

### Distribution analysis
Outputs a table of the amount of simulated Yatzy games which achieved a specific point total `dist_scores` as well as the overall measured probabilities of fulfilling the requirements of a space `dist_categories`.

```bash
python YatzySuite.py --n 1000000
```

### Deviation study
Outputs a table of the average deviation in percentage points between measured probabilities for games of a given size and true value of fulfilling the requirements of a space `study_summary`, and a full table of the individual probabilties in all simulated games `study_deviation`.

Default reps: 5

```bash
python YatzySuite.py --study 1000,10000,100000 --reps 10
```

### Custom output directory
Default output: results

```bash
python YatzySuite.py --n 500000 --output my_results
```

### Output Files

```
results/
  dist_scores_*.csv
  dist_categories_*.csv
  study_deviation_*.csv
  study_summary_*.csv
  meta_*.json
```
Where * is replaced with the time (seconds since the epoch) the simulation was initiated.
