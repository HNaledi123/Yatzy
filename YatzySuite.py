# --- DEPENDENCY CHECKS ---
import sys

dependencies = ["numpy", "numba"]
for dep in dependencies:
    try:
        __import__(dep)
    except ImportError:
        sys.exit(f"Error: {dep} is required. Please install it using 'pip install {dep}'.")

# --- IMPORTS ---

import argparse
import csv
import json
import traceback
import os
import time
import sys
import concurrent.futures
import numpy as np
from itertools import product
from typing import Tuple, Optional
from pathlib import Path
from numba import njit

# --- CONFIGURATION & CONSTANTS ---

CATEGORY_NAMES = [
    "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Three of a Kind", "Four of a Kind", "Yatzy",
    "Two Pairs", "Small Straight", "Large Straight", "Full House", "Chance"
]

NUM_CATEGORIES = 15
DICE_COUNT = 5
DICE_FACES = 6
ROLL_STATE_COUNT = DICE_FACES ** DICE_COUNT
MAX_SCORE = 374
SCORE_BINS_SIZE = MAX_SCORE + 1
YATZY_INDEX = 9 # "Yatzy" is tenth in the CATEGORY_NAMES array.

# Exact probabilities based on 7776 possible outcomes (6^5)
EXPECTED_PROBS = np.array([
    4651 / ROLL_STATE_COUNT,    # Aces
    4651 / ROLL_STATE_COUNT,    # Twos
    4651 / ROLL_STATE_COUNT,    # Threes
    4651 / ROLL_STATE_COUNT,    # Fours
    4651 / ROLL_STATE_COUNT,    # Fives
    4651 / ROLL_STATE_COUNT,    # Sixes
    7056 / ROLL_STATE_COUNT,    # One Pair
    1656 / ROLL_STATE_COUNT,    # Three of a Kind
    156  / ROLL_STATE_COUNT,    # Four of a Kind
    6    / ROLL_STATE_COUNT,    # Yatzy
    2100 / ROLL_STATE_COUNT,    # Two Pairs
    120  / ROLL_STATE_COUNT,    # Small Straight
    120  / ROLL_STATE_COUNT,    # Large Straight
    300  / ROLL_STATE_COUNT,    # Full House
    1.0                         # Chance
])

# --- LOOKUP TABLE GENERATION ---

def _build_lookup_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates lookup tables for all possible dice roll outcomes.

    This function pre-calculates and caches:
    - scores: The score for each category for every possible dice roll.
    - satisfaction_mask: A boolean mask indicating if a category is achieved (satisifed).
    - keep_value: The face value of the dice to keep for the re-roll strategy.
    - keep_count: The count of dice to keep for the re-roll strategy.
    - priority: An array defining the preferred order for choosing categories.

    Returns:
        tuple: A tuple containing the generated np arrays:
               (scores, satisfaction_mask, keep_value, keep_count, priority).
    """
    scores = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
    satisfaction_mask = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    keep_value = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    keep_count = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)

    # In case of tie, prefer earlier entries in this array
    # IDs correspond to the order in which they were named in CATEGORY_NAMES
    priority = np.array([
        0,  # Ettor
        1,  # Tvåor
        2,  # Treor
        3,  # Fyror
        4,  # Femmor
        5,  # Sexor
        6,  # Ett par
        7,  # Tretal
        8,  # Fyrtal
        9,  # Yatzy
        10, # Two Pairs
        11, # Small Straight
        12, # Large Straight
        13, # Full House
        14  # Chance
    ], dtype=np.int8)

    for index, dice in enumerate(product(range(1, DICE_FACES + 1), repeat=DICE_COUNT)):
        dice_array = np.array(dice, dtype=np.int8)
        counts = np.bincount(dice_array, minlength=7) # Counts[0] unused

        # Upper Section (Aces through Sixes)
        for face in range(1, 7):
            scores[index, face-1] = counts[face] * face
            if counts[face] > 0: satisfaction_mask[index, face-1] = 1
        
        # N of a Kind (2-5)
        for count in range(2, 6):
            if count == 5:
                if np.any(counts[1:] >= 5):
                    scores[index, count+4] = 50
                    satisfaction_mask[index, count+4] = 1
            else:
                for face in range(6, 0, -1):
                    if counts[face] >= count:
                        scores[index, count+4] = face * count
                        satisfaction_mask[index, count+4] = 1
                        break
        
        # Two Pairs
        pairs = [face for face in range(1, 7) if counts[face] >= 2]
        if len(pairs) >= 2:
            scores[index, 10] = pairs[-1]*2 + pairs[-2]*2
            satisfaction_mask[index, 10] = 1
            
        # Straights
        unique_values = np.unique(dice_array)
        if np.array_equal(unique_values, [1,2,3,4,5]): 
            scores[index, 11] = 15
            satisfaction_mask[index, 11] = 1
        if np.array_equal(unique_values, [2,3,4,5,6]): 
            scores[index, 12] = 20
            satisfaction_mask[index, 12] = 1
            
        # Full House
        if np.any(counts == 3) and np.any(counts == 2):
            scores[index, 13] = (np.where(counts==3)[0][0]*3 + np.where(counts==2)[0][0]*2)
            satisfaction_mask[index, 13] = 1

        # Chance
        scores[index, 14] = dice_array.sum()
        satisfaction_mask[index, 14] = 1

        # Pre-calculate the keep strategy: find the face value with the highest count.
        max_count = counts.max()
        keep_value[index] = max(face for face in range(1, 7) if counts[face] == max_count)
        keep_count[index] = max_count

    return scores, satisfaction_mask, keep_value, keep_count, priority

print("Building lookup tables...", end="", flush=True)
LOOKUP_SCORES, LOOKUP_SATISFACTION_MASK, LOOKUP_KEEP_VALUE, LOOKUP_KEEP_COUNT, LOOKUP_PRIORITY = _build_lookup_tables()
print(" Done.")

# --- NUMBA FUNCTIONS ---

@njit(nogil=True)
def _encode(dice: np.ndarray) -> int:
    """
    Encodes a 5-dice roll into a unique integer index from 0 to 7775.

    Args:
        dice (np.ndarray): A np array of 5 integers (1-6) representing the dice.

    Returns:
        int: The unique integer index for the roll.
    """
    k = 0
    for i in range(DICE_COUNT): 
        k = k * 6 + (dice[i] - 1)
    return k

@njit(nogil=True)
def _reroll(dice: np.ndarray, roll_index: int, out: np.ndarray) -> None:
    """
    Determines which dice to re-roll based on a simple strategy.
    The strategy is to keep the dice that appear most frequently.
    The new roll (with some dice kept, some re-rolled) is placed in `out`.
    """
    cnt = LOOKUP_KEEP_COUNT[roll_index]
    if cnt == 5:
        for i in range(DICE_COUNT):
            out[i] = dice[i]
        return
        
    val = LOOKUP_KEEP_VALUE[roll_index]
    for i in range(cnt): 
        out[i] = val
    for i in range(cnt, DICE_COUNT): 
        out[i] = np.random.randint(1, 7)

@njit(nogil=True)
def _play_game(stats0: np.ndarray, stats1: np.ndarray, stats2: np.ndarray, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> Tuple[int, bool, bool]:
    """
    Simulates a single, complete game of Yatzy using an optimized, Numba-jitted function.

    This function handles the 15 rounds of a game, including dice rolls,
    AI-driven re-rolls, and category selection based on a fixed priority.
    It also tracks statistics about which categories were achievable on each roll.

    Args:
        stats0 (np.ndarray): Array to accumulate category hits on the 1st roll.
        stats1 (np.ndarray): Array to accumulate category hits on the 2nd roll.
        stats2 (np.ndarray): Array to accumulate category hits on the 3rd roll.
        d0 (np.ndarray): Pre-allocated array for the 1st dice roll.
        d1 (np.ndarray): Pre-allocated array for the 2nd dice roll.
        d2 (np.ndarray): Pre-allocated array for the 3rd dice roll.

    Returns:
        tuple: A tuple containing (final_score, has_bonus, has_yatzy).
    """
    # Bitmask for available categories. 0x7FFF = (1 << 15) - 1
    allowed_categories = 0x7FFF 
    total = 0
    upper = 0
    got_yatzy = False
    
    # A game consists of 15 rounds, one for each category
    for _ in range(NUM_CATEGORIES): 
        # Roll 1
        for i in range(DICE_COUNT): 
            d0[i] = np.random.randint(1, 7)
        index0 = _encode(d0)
        
        row0 = LOOKUP_SATISFACTION_MASK[index0]
        for c in range(NUM_CATEGORIES): 
            stats0[c] += row0[c]
        
        # Roll 2
        _reroll(d0, index0, d1)
        index1 = _encode(d1)
        row1 = LOOKUP_SATISFACTION_MASK[index1]
        for c in range(NUM_CATEGORIES): 
            stats1[c] += row1[c]
        
        # Roll 3
        _reroll(d1, index1, d2)
        index2 = _encode(d2)
        row2 = LOOKUP_SATISFACTION_MASK[index2]
        for c in range(NUM_CATEGORIES): 
            stats2[c] += row2[c]
        
        best_sc = -1
        best_id = -1
        scores = LOOKUP_SCORES[index2]
        
        for i in range(NUM_CATEGORIES):
            cat = LOOKUP_PRIORITY[i]
            if (allowed_categories >> cat) & 1:
                s = scores[cat]
                if s > best_sc:
                    best_sc = s
                    best_id = cat
        
        total += best_sc
        if best_id < 6: 
            upper += best_sc
        if best_id == YATZY_INDEX and best_sc > 0: 
            got_yatzy = True
        
        allowed_categories &= ~(1 << best_id)
        
    bonus = 50 if upper >= 63 else 0
    return total + bonus, bonus > 0, got_yatzy

@njit(nogil=True)
def _worker_sim_batch(count: int, seed: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Worker function to simulate a batch of Yatzy games and return aggregated results.

    This function is designed to be run in a separate thread. It initializes
    its own random seed and runs a specified number of games, aggregating
    the results and statistics locally to ensure constant memory usage.

    Args:
        count (int): The number of games to simulate in this batch.
        seed (int): The random seed to initialize for this worker.

    Returns:
        tuple: Aggregated results: (total_score, score_bins, bins_ny_nb,
               bins_ny_yb, bins_yy_nb, bins_yy_yb, stats_roll1, stats_roll2, stats_roll3).
    """
    np.random.seed(seed)
    # Local aggregates
    total_score = 0
    score_bins = np.zeros(SCORE_BINS_SIZE, dtype=np.uint32)
    bins_ny_nb = np.zeros(SCORE_BINS_SIZE, dtype=np.uint32)
    bins_ny_yb = np.zeros(SCORE_BINS_SIZE, dtype=np.uint32)
    bins_yy_nb = np.zeros(SCORE_BINS_SIZE, dtype=np.uint32)
    bins_yy_yb = np.zeros(SCORE_BINS_SIZE, dtype=np.uint32)
    local_stats_roll1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    local_stats_roll2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    local_stats_roll3 = np.zeros(NUM_CATEGORIES, dtype=np.int64)

    d0 = np.empty(DICE_COUNT, dtype=np.int8)
    d1 = np.empty(DICE_COUNT, dtype=np.int8)
    d2 = np.empty(DICE_COUNT, dtype=np.int8)

    for i in range(count):
        score, has_bonus, has_yatzy = _play_game(local_stats_roll1, local_stats_roll2, local_stats_roll3, d0, d1, d2)
        total_score += score
        score_bins[score] += 1
        if not has_yatzy and not has_bonus:
            bins_ny_nb[score] += 1
        elif not has_yatzy and has_bonus:
            bins_ny_yb[score] += 1
        elif has_yatzy and not has_bonus:
            bins_yy_nb[score] += 1
        else:  # has_yatzy and has_bonus
            bins_yy_yb[score] += 1

    return total_score, score_bins, bins_ny_nb, bins_ny_yb, bins_yy_nb, bins_yy_yb, local_stats_roll1, local_stats_roll2, local_stats_roll3

# --- DRIVER LOGIC ---

def _format_time(seconds: float) -> str:
    """Formats seconds into a HH:MM:SS or MM:SS string."""
    if seconds is None or seconds < 0 or seconds == float('inf'):
        return "--:--"

    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _print_batch_progress(elapsed: float, games_completed: int, total_games: int, prefix: str = "Simulating") -> None:
    """
    Prints progress information for batch simulations.
    
    Args:
        elapsed (float): Elapsed time in seconds.
        games_completed (int): Number of games completed so far.
        total_games (int): Total number of games to complete.
        prefix (str): Prefix for the progress message.
    """
    rate = games_completed / elapsed if elapsed > 0 else 0
    eta = (float(total_games) - games_completed) / rate if rate > 0 else 0
    pct = games_completed / total_games * 100
    eta_str = _format_time(eta)
    if "Study" in prefix:
        print(f"\r{prefix}: {pct:5.1f}% | {games_completed:,}/{total_games:,} games | ETA: {eta_str}".ljust(80), end="", flush=True)
    else:
        print(f"\r{prefix}: {pct:5.1f}% | {rate:9,.0f} games/s | ETA: {eta_str}".ljust(80), end="", flush=True)

def run_simulation_parallel(total_count: int, batch_size: Optional[int] = None, quiet: bool = False) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs Yatzy simulations in parallel using a streaming, constant-memory approach.

    This function uses a thread pool with a bounded number of in-flight tasks.
    Workers aggregate results locally, and the main thread merges these aggregates
    as they complete, ensuring that memory usage does not scale with `total_count`.

    Args:
        total_count (int): The total number of games to simulate.
        batch_size (int, optional): The number of games per worker batch.
                                    If None, a suitable size is calculated.
        quiet (bool, optional): If True, suppress progress output. Defaults to False.

    Returns:
        tuple: Final aggregated results from all workers.
    """    
    cpu_count = os.cpu_count() or 4
    if batch_size is None:
        target_chunks = cpu_count * 4
        batch_size = max(1000, total_count // target_chunks)
        batch_size = min(batch_size, 100_000)

    # Global aggregates
    agg_total_score = 0
    agg_score_bins = np.zeros(SCORE_BINS_SIZE, dtype=object)
    agg_bins_ny_nb = np.zeros(SCORE_BINS_SIZE, dtype=object)
    agg_bins_ny_yb = np.zeros(SCORE_BINS_SIZE, dtype=object)
    agg_bins_yy_nb = np.zeros(SCORE_BINS_SIZE, dtype=object)
    agg_bins_yy_yb = np.zeros(SCORE_BINS_SIZE, dtype=object)
    aggregate_stats_roll1 = np.zeros(NUM_CATEGORIES, dtype=object)
    aggregate_stats_roll2 = np.zeros(NUM_CATEGORIES, dtype=object)
    aggregate_stats_roll3 = np.zeros(NUM_CATEGORIES, dtype=object)

    start_time = time.time()
    max_in_flight = cpu_count * 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = set()
        sims_submitted = 0 
        sims_completed = 0 

        while sims_completed < total_count:
            # Submit new tasks only if there is capacity
            while sims_submitted < total_count and len(futures) < max_in_flight:
                count = min(batch_size, total_count - sims_submitted)
                seed = np.random.randint(0, 2**30)
                futures.add(executor.submit(_worker_sim_batch, count, seed))
                sims_submitted += count

            # Wait for the next future to complete
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for f in done:
                try:
                    # Retrieve and aggregate results immediately
                    (total_score, score_bins, ny_nb, ny_yb, yy_nb, yy_yb, s0, s1, s2) = f.result()
                    agg_total_score += total_score
                    agg_score_bins += score_bins
                    agg_bins_ny_nb += ny_nb
                    agg_bins_ny_yb += ny_yb
                    agg_bins_yy_nb += yy_nb
                    agg_bins_yy_yb += yy_yb
                    aggregate_stats_roll1 += s0
                    aggregate_stats_roll2 += s1
                    aggregate_stats_roll3 += s2
                    sims_completed += int(sum(score_bins))
                except Exception as e:
                    print("\n\n--- A CRITICAL ERROR OCCURRED ---")
                    print(f"Simulation halted due to an exception in a worker thread: {e}")
                    traceback.print_exc()
                    print("--- END OF ERROR REPORT ---")
                    raise e

            elapsed = time.time() - start_time
            if not quiet:
                _print_batch_progress(elapsed, sims_completed, total_count, "Simulating")

    if not quiet:
        print()
    return agg_total_score, agg_score_bins, agg_bins_ny_nb, agg_bins_ny_yb, agg_bins_yy_nb, agg_bins_yy_yb, aggregate_stats_roll1, aggregate_stats_roll2, aggregate_stats_roll3

# --- MAIN EXECUTION ---

def run_suite(args: argparse.Namespace) -> None:
    """
    Main entry point to run the simulation suite based on command-line arguments.

    This function orchestrates the different modes of operation:
    1. Distribution Analysis: Runs a large number of simulations and saves score
       and category distributions.
    2. Probability Deviation Study: Runs simulations at different sample sizes
       to study the deviation of observed probabilities from theoretical ones.
    """
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    
    # Mode 1: Distribution Analysis
    if args.n:
        print(f"\n=== MODE 1: DISTRIBUTION ANALYSIS ({args.n:,} sim) ===")
        start_t = time.time()
        total_score, score_bins, bins_ny_nb, bins_ny_yb, bins_yy_nb, bins_yy_yb, stats_roll1, stats_roll2, stats_roll3 = run_simulation_parallel(args.n)
        print("Simulations complete. Processing data...")

        with open(out_dir / f"dist_scores_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Score", "Count_Total", "NoYatzy_NoBonus", "NoYatzy_YesBonus", "YesYatzy_NoBonus", "YesYatzy_YesBonus"])
            for s in range(SCORE_BINS_SIZE):
                if score_bins[s] > 0:
                    w.writerow([s, int(score_bins[s]), int(bins_ny_nb[s]), int(bins_ny_yb[s]), int(bins_yy_nb[s]), int(bins_yy_yb[s])])

        tot_r = args.n * NUM_CATEGORIES
        with open(out_dir / f"dist_categories_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Roll1_Hits", "Roll1_Prob", "Roll2_Hits", "Roll2_Prob", "Roll3_Hits", "Roll3_Prob"])
            for i, cat in enumerate(CATEGORY_NAMES):
                w.writerow([cat, stats_roll1[i], stats_roll1[i]/tot_r, stats_roll2[i], stats_roll2[i]/tot_r, stats_roll3[i], stats_roll3[i]/tot_r])
                
        elapsed_tot = time.time() - start_t
        meta = {
            "mode": "distribution",
            "count": args.n,
            "elapsed_sec": elapsed_tot,
            "mean_score": total_score / args.n if args.n > 0 else 0,
            "performance": f"{args.n/elapsed_tot:.0f} games/sec"
        }
        with open(out_dir / f"meta_dist_{ts}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Done! Data saved to {out_dir}")

    # Mode 2: Probability Deviation Study
    if args.study:
        steps = [int(x) for x in args.study.split(",")]
        reps = args.reps
        print(f"\n=== MODE 2: DEVIATION STUDY ({len(steps)} steps, {reps} reps/step) ===")
        _ = run_simulation_parallel(1, batch_size=1, quiet=True)
        results = []
        summary_data = {cat: {step: [] for step in steps} for cat in CATEGORY_NAMES}
        timing_data = {step: [] for step in steps}
        
        total_games_in_study = sum(steps) * reps
        games_completed = 0
        start_t_study = time.time()
        
        for i_step, step in enumerate(steps):
            for r in range(reps):
                print(f"\nRunning Step {i_step+1}/{len(steps)} (size: {step:,}), Rep {r+1}/{reps}...")
                step_start_time = time.time()
                study_batch_size = min(100_000, max(1000, step//(os.cpu_count() or 4)))
                _, _, _, _, _, _, stats_roll1, _, _ = run_simulation_parallel(step, batch_size=study_batch_size)
                step_elapsed = time.time() - step_start_time
                timing_data[step].append(step_elapsed)
                
                games_completed += step
                total_rolls = step * NUM_CATEGORIES
                obs_probs = stats_roll1 / total_rolls
                abs_devs = np.abs(obs_probs - EXPECTED_PROBS) * 100 
                
                for i, cat in enumerate(CATEGORY_NAMES):
                    dev_val = abs_devs[i]
                    results.append({
                        "Simulations": step,
                        "Repetition": r+1,
                        "Category": cat,
                        "Expected_Pct": EXPECTED_PROBS[i] * 100,
                        "Observed_Pct": obs_probs[i] * 100,
                        "Abs_Deviation_Pct": dev_val
                    })
                    summary_data[cat][step].append(dev_val)
                
                elapsed = time.time() - start_t_study
                _print_batch_progress(elapsed, games_completed, total_games_in_study, "Overall Study Progress")

        print("\nSimulations complete. Processing data...")
        with open(out_dir / f"study_deviation_{ts}.csv", "w", newline="") as f:
            fields = ["Simulations", "Repetition", "Category", "Expected_Pct", "Observed_Pct", "Abs_Deviation_Pct"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)

        with open(out_dir / f"study_summary_{ts}.csv", "w", newline="") as f:
            headers = ["Category"] + [f"Sim_{s}" for s in steps]
            w = csv.writer(f)
            w.writerow(headers)
            for cat in CATEGORY_NAMES:
                row = [cat]
                for step in steps:
                    devs = summary_data[cat][step]
                    avg_dev = sum(devs) / len(devs) if devs else 0
                    row.append(f"{avg_dev:.4g}")
                w.writerow(row)

        meta_study = {
            "mode": "deviation_study",
            "steps": steps,
            "reps": reps,
            "elapsed_sec": time.time() - start_t_study,
            "timing_stats": {
                str(step): {
                    "avg_sec": sum(timing_data[step]) / len(timing_data[step]),
                    "min_sec": min(timing_data[step]),
                    "max_sec": max(timing_data[step])
                }
                for step in steps
            }
        }
        with open(out_dir / f"meta_study_{ts}.json", "w") as f:
            json.dump(meta_study, f, indent=2)
        print(f"Done! Data saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yatzy Suite: High-Performance Simulation")
    parser.add_argument("--n", type=int, help="Number of simulations for distribution analysis")
    parser.add_argument("--study", type=str, help="Comma-separated list of simulation steps")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per step in study mode")
    parser.add_argument("--output", type=str, default="results", help="Output directory")

    args = parser.parse_args()
    
    if not args.n and not args.study:
        parser.print_help()
    else:
        run_suite(args)