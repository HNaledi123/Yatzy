import argparse
import csv
import json
import traceback
import os
import time
import sys
import concurrent.futures
from typing import Tuple, Optional
from pathlib import Path

# --- DEPENDENCY CHECKS ---
try:
    import numpy as np
except ImportError:
    sys.exit("Error: Numpy is required. Please install it using 'pip install numpy'.")

try:
    from numba import njit
except ImportError:
    sys.exit("Error: Numba is required. Please install it using 'pip install numba'.")

# Promote numpy warnings (like overflow) to exceptions to halt execution
np.seterr(all='raise')

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
MAX_SCORE = 374 # Theoretical maximum score in Yatzy
SCORE_BINS_SIZE = MAX_SCORE + 1
# Mapping strict logic indices
OF_A_KIND_TO_IDX = {2: 6, 3: 7, 4: 8, 5: 9}
YATZY_IDX = 9

# Exact probabilities based on 7776 outcomes (6^5)
EXPECTED_PROBS = np.array([
    4651 / ROLL_STATE_COUNT,  # Aces
    4651 / ROLL_STATE_COUNT,  # Twos
    4651 / ROLL_STATE_COUNT,  # Threes
    4651 / ROLL_STATE_COUNT,  # Fours
    4651 / ROLL_STATE_COUNT,  # Fives
    4651 / ROLL_STATE_COUNT,  # Sixes
    7056 / ROLL_STATE_COUNT,  # One Pair
    1656 / ROLL_STATE_COUNT,  # Three of a Kind
    156  / ROLL_STATE_COUNT,  # Four of a Kind
    6    / ROLL_STATE_COUNT,  # Yatzy
    2100 / ROLL_STATE_COUNT,  # Two Pairs
    120  / ROLL_STATE_COUNT,  # Small Straight
    120  / ROLL_STATE_COUNT,  # Large Straight
    300  / ROLL_STATE_COUNT,  # Full House
    1.0         # Chance
])

# --- LOOKUP TABLE GENERATION ---

def _build_lookup_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates lookup tables for all possible 7776 dice roll outcomes (6^5).

    This function pre-calculates and caches:
    - scores: The score for each category for every possible dice roll.
    - sat_mask: A boolean mask indicating if a category is satisfied (achieved).
    - keep_val: The face value of the dice to keep for the AI re-roll strategy.
    - keep_cnt: The count of dice to keep for the AI re-roll strategy.
    - priority: An array defining the AI's preferred order for choosing categories.

    Returns:
        tuple: A tuple containing the generated numpy arrays:
               (scores, sat_mask, keep_val, keep_cnt, priority).
    """
    from itertools import product
    scores = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
    sat_mask = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    keep_val = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    keep_cnt = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)

    # This array defines the order in which the AI checks for available categories to score
    priority = np.array([
        5,  # Sixes
        4,  # Fives
        3,  # Fours
        2,  # Threes
        1,  # Twos
        0,  # Aces
        9,  # Yatzy
        8,  # Four of a Kind
        7,  # Three of a Kind
        6,  # One Pair
        10, # Two Pairs
        12, # Large Straight
        11, # Small Straight
        13, # Full House
        14  # Chance (fallback)
    ], dtype=np.int8)

    for idx, dice in enumerate(product(range(1, DICE_FACES + 1), repeat=DICE_COUNT)):
        d_arr = np.array(dice, dtype=np.int8)
        counts = np.bincount(d_arr, minlength=7)
        s_dice = np.sort(d_arr)
        d_sum = d_arr.sum()

        # Upper Section (Aces through Sixes)
        for f in range(1, 7):
            scores[idx, f-1] = counts[f] * f
            if counts[f] > 0: sat_mask[idx, f-1] = 1
        
        # X of a Kind
        for k, p_idx in OF_A_KIND_TO_IDX.items():
            if k == 5: 
                if np.any(counts[1:] >= 5):
                    scores[idx, p_idx] = 50
                    sat_mask[idx, p_idx] = 1
            else:
                for f in range(6, 0, -1):
                    if counts[f] >= k:
                        scores[idx, p_idx] = f * k
                        sat_mask[idx, p_idx] = 1
                        break
        
        # Two Pairs
        pairs = [f for f in range(1, 7) if counts[f] >= 2]
        if len(pairs) >= 2:
            scores[idx, 10] = pairs[-1]*2 + pairs[-2]*2
            sat_mask[idx, 10] = 1
            
        # Straights
        u_vals = np.unique(s_dice)
        if np.array_equal(u_vals, [1,2,3,4,5]): 
            scores[idx, 11] = 15
            sat_mask[idx, 11] = 1
        if np.array_equal(u_vals, [2,3,4,5,6]): 
            scores[idx, 12] = 20
            sat_mask[idx, 12] = 1
            
        # Full House
        if np.any(counts == 3) and np.any(counts == 2):
            # Score is the sum of all dice, which is correct for a Full House.
            scores[idx, 13] = (np.where(counts==3)[0][0]*3 + np.where(counts==2)[0][0]*2)
            sat_mask[idx, 13] = 1

        # Chance
        scores[idx, 14] = d_sum
        sat_mask[idx, 14] = 1

        # Pre-calc keep strategy: find the face value with the highest count.
        c_no_zero = counts[1:]
        mx = c_no_zero.max()
        kv = 1
        for f in range(6, 0, -1):
            if counts[f] == mx:
                kv = f
                break
        keep_val[idx] = kv
        keep_cnt[idx] = mx

    return scores, sat_mask, keep_val, keep_cnt, priority

print("Building lookup tables...", end="", flush=True)
TBL_SCORES, TBL_SAT, TBL_KEEP_VAL, TBL_KEEP_CNT, TBL_PRIO = _build_lookup_tables()
print("Done.")

# --- OPTIMIZED NUMBA FUNCTIONS ---

@njit(nogil=True)
def _encode(dice: np.ndarray) -> int:
    """
    Encodes a 5-dice roll into a unique integer index from 0 to 7775.

    This treats the dice roll as a base-6 number, allowing it to be used
    as an index into the pre-calculated lookup tables.
    e.g., (d1-1)*6^4 + (d2-1)*6^3 + ... + (d5-1)*6^0

    Args:
        dice (np.ndarray): A numpy array of 5 integers (1-6) representing the dice.

    Returns:
        int: The unique integer index for the roll.
    """
    k = 0
    for i in range(DICE_COUNT): 
        k = k * 6 + (dice[i] - 1)
    return k

@njit(nogil=True)
def _ai_reroll(dice: np.ndarray, roll_idx: int, out: np.ndarray) -> None:
    """
    Determines which dice to re-roll based on a simple AI strategy.
    The strategy is to keep the dice that appear most frequently.
    The new roll (with some dice kept, some re-rolled) is placed in `out`.
    """
    cnt = TBL_KEEP_CNT[roll_idx]
    if cnt == 5:
        for i in range(DICE_COUNT):
            out[i] = dice[i]
        return
        
    val = TBL_KEEP_VAL[roll_idx]
    for i in range(cnt): 
        out[i] = val
    for i in range(cnt, DICE_COUNT): 
        out[i] = np.random.randint(1, 7)

@njit(nogil=True)
def _play_game_optimized(stats0: np.ndarray, stats1: np.ndarray, stats2: np.ndarray, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> Tuple[int, bool, bool]:
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
        idx0 = _encode(d0)
        
        row0 = TBL_SAT[idx0]
        for c in range(NUM_CATEGORIES): 
            stats0[c] += row0[c]
        
        # Roll 2
        _ai_reroll(d0, idx0, d1)
        idx1 = _encode(d1)
        row1 = TBL_SAT[idx1]
        for c in range(NUM_CATEGORIES): 
            stats1[c] += row1[c]
        
        # Roll 3
        _ai_reroll(d1, idx1, d2)
        idx2 = _encode(d2)
        row2 = TBL_SAT[idx2]
        for c in range(NUM_CATEGORIES): 
            stats2[c] += row2[c]
        
        best_sc = -1
        best_id = -1
        scores = TBL_SCORES[idx2]
        
        for i in range(NUM_CATEGORIES):
            cat = TBL_PRIO[i]
            if (allowed_categories >> cat) & 1:
                s = scores[cat]
                if s > best_sc:
                    best_sc = s
                    best_id = cat
        
        total += best_sc
        if best_id < 6: 
            upper += best_sc
        if best_id == YATZY_IDX and best_sc > 0: 
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
        score, has_bonus, has_yatzy = _play_game_optimized(local_stats_roll1, local_stats_roll2, local_stats_roll3, d0, d1, d2)
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
    # Include game count only if the prefix indicates it's a study
    if "Study" in prefix:
        print(f"\r{prefix}: {pct:5.1f}% | {games_completed:,}/{total_games:,} games | ETA: {eta:.0f}s ", end="", flush=True)
    else:
        print(f"\r{prefix}: {pct:5.1f}% | {rate:9,.0f} games/s | ETA: {eta:3.0f}s ", end="", flush=True)

def run_simulation_parallel(total_count: int, batch_size: Optional[int] = None, show_progress: bool = True) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs Yatzy simulations in parallel using a streaming, constant-memory approach.

    This function uses a thread pool with a bounded number of in-flight tasks.
    Workers aggregate results locally, and the main thread merges these aggregates
    as they complete, ensuring that memory usage does not scale with `total_count`.

    Args:
        total_count (int): The total number of games to simulate.
        batch_size (int, optional): The number of games per worker batch.
                                    If None, a suitable size is calculated.
        show_progress (bool): Whether to print progress updates.

    Returns:
        tuple: Final aggregated results from all workers.
    """    
    cpu_count = os.cpu_count() or 4
    if batch_size is None:
        target_chunks = cpu_count * 4
        batch_size = max(1000, total_count // target_chunks)
    batch_size = min(batch_size, 100_000)

    # Global aggregates
    agg_total_score = 0 # Use Python int for arbitrary precision
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
        sims_submitted = 0 # Use Python int
        sims_completed = 0 # Use Python int

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
                    # We must re-raise to stop the program
                    raise e

            if show_progress:
                elapsed = time.time() - start_time
                _print_batch_progress(elapsed, sims_completed, total_count, "Simulating")

    if show_progress:
        print() # Newline after loop
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
        print("Processing data...")

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
        print("Warming up JIT compiler...", end="", flush=True)
        _ = run_simulation_parallel(1, batch_size=1)
        print(" Done.")
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
                _, _, _, _, _, _, stats_roll1, _, _ = run_simulation_parallel(step, batch_size=max(1000, step//(os.cpu_count() or 1)))
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

        print("Study simulations complete. Saving data...")
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
        print(f"Study saved to {out_dir}")

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