# Monte Carlo Yatzy Simulator

High-performance Monte Carlo simulations for **Scandinavian Yatzy**, focused on score distributions, category probabilities, and convergence behavior at large sample sizes.

## Usage

Run the suite in one of two primary modes: distribution analysis (`--n`) or deviation study (`--study`). Output files are written into the directory specified with `--output` (default: `results`).

Examples:

```bash
python YatzySuite.py --n 1000000
python YatzySuite.py --study 1000,10000,100000 --reps 10
python YatzySuite.py --n 500000 --output my_results
```

## CLI Arguments

- `--n`: Number of simulations for distribution analysis (int). No default — required for distribution mode.
- `--study`: Comma-separated list of simulation sizes for the deviation study (str). No default — required for study mode.
- `--reps`: Repetitions per step in study mode (int). Default: `5`.
- `--output`: Output directory where results are written (str). Default: `results`.
- `--seed`: Base seed for deterministic runs (int). Default: `12345`.
- `--threads`: Number of worker threads to use (int). Default: system CPU count (automatic when omitted).
- `--batch-size` / `--batch_size`: Batch size per worker (int). Default: automatic selection (the code chooses a suitable batch size when omitted).

Notes:
- If neither `--n` nor `--study` are provided the script prints help and exits.
- `--batch-size` overrides the automatic batch size calculation if provided.

## Modes & Outputs

### Distribution analysis (`--n`)

Generates per-score counts and category hit probabilities across `n` simulated games. Files produced:

```
dist_scores_<ts>.csv         # Score counts (per-score)
dist_categories_<ts>.csv    # Category hit counts & probabilities
dist_scores_<ts>.png        # Score distribution plot
dist_groups_<ts>.png        # Grouped distributions by bonus/yatzy outcome
dist_summary_<ts>.csv       # Summary statistics (mean, sd, percentiles)
meta_dist_<ts>.json         # Run metadata and timing
```

### Deviation study (`--study`)

Runs simulations at multiple sizes and records deviations between observed and expected category probabilities. Files produced:

```
study_deviation_<ts>.csv    # Raw deviations per repetition
study_summary_<ts>.csv      # Average deviations per category per step
study_deviation_plot_<ts>.png
meta_study_<ts>.json        # Run metadata and timing
```

`<ts>` in filenames is a timestamp (seconds since the epoch) created when the run starts.

## Output directory

The specified `--output` directory is created if it does not exist. Default: `results`.

## Notes on determinism and performance

- Use `--seed` to produce repeatable runs across machines (where available).
- `--threads` can be used to limit CPU usage; by default the script uses all available logical CPUs.
- `--batch-size` is provided for advanced tuning; in most cases, letting the script choose is recommended.

## Requirements

- Python 3.10+ (recommended)
- numpy, numba, matplotlib

Run `pip install numpy numba matplotlib` if needed.
