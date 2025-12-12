"""
Yatzy Suite: A High-Performance Monte Carlo Simulation Tool

This script is designed to run a large number of simulated Yatzy games to analyze
statistical outcomes, such as score distributions and the probability of achieving
certain categories. It serves as the data generation engine for the research report.

Methodology:
The simulation employs the Monte Carlo method. Each game is played by a simple,
deterministic AI agent. To achieve high performance suitable for millions of
simulations, the script heavily relies on two key optimizations:

1.  Lookup Tables: All 7,776 possible outcomes of a 5-dice roll are pre-calculated
    at startup. This includes scores for every category and a basic re-roll strategy.
2.  Numba JIT Compilation: The core game-playing logic is Just-In-Time compiled
    using Numba, which translates Python code into highly efficient machine code.
"""
import argparse
import csv
import json
import os
import time
import sys
import concurrent.futures
from typing import Tuple
from pathlib import Path

# --- DEPENDENCY CHECKS ---
try:
    import numpy as np
except ImportError:
    sys.exit("Error: Numpy is required. Please install it using 'pip install numpy'.")

try:
    from numba import njit, prange
except ImportError:
    sys.exit("Error: Numba is required. Please install it using 'pip install numba'.")

# --- CONFIGURATION & CONSTANTS ---

CATEGORY_NAMES: list[str] = [
    "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Three of a Kind", "Four of a Kind", "Yatzy",
    "Two Pairs", "Small Straight", "Large Straight", "Full House", "Chance"
]

NUM_CATEGORIES = 15
# Mapping strict logic indices
OF_A_KIND_TO_IDX: dict[int, int] = {2: 6, 3: 7, 4: 8, 5: 9}
YATZY_IDX = 9
ROLL_STATE_COUNT = 7776 # 6^5

# Game Rule Constants
UPPER_SECTION_BONUS_THRESHOLD = 63
UPPER_SECTION_BONUS_SCORE = 50
YATZY_SCORE = 50
SMALL_STRAIGHT_SCORE = 15 # 1 + 2 + 3 + 4 + 5
LARGE_STRAIGHT_SCORE = 20 # 2 + 3 + 4 + 5 + 6
ALL_CATEGORIES_MASK = (1 << NUM_CATEGORIES) - 1 # Creates a mask with 15 ones in binary

# Exact probabilities of category requirements being fulfilled based on 7776 outcomes (6^5)
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
    Pre-calculates all possible outcomes for a single 5-dice roll.

    This function iterates through all 7,776 (6^5) combinations of 5 dice. For each
    combination, it assigns an id and then calculates the score for all 15 Yatzy categories 
    and determines a basic re-roll strategy. This avoids costly recalculations during the simulation.

    Returns:
        A tuple containing five numpy arrays:
        - scores: (7776, 15) array of scores for each roll and category.
        - sat_mask: (7776, 15) boolean mask indicating if a roll satisfies a category.
        - keep_val: (7776,) value of the dice to keep.
        - keep_cnt: (7776,) amount of dice with the selected value.
        - priority: (15,) array defining the category selection order. Lower is prioritised.
    """
    from itertools import product
    scores = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
    sat_mask = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    keep_val = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    keep_cnt = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    
    # Priority for AI decision making
    priority = np.array([5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 10, 12, 11, 13, 14], dtype=np.int8)

    for idx, dice in enumerate(product(range(1, 7), repeat=5)):
        d_arr = np.array(dice, dtype=np.int8)
        counts = np.bincount(d_arr, minlength=7)
        d_sum = d_arr.sum()

        # Upper Section (Aces through Sixes)
        for f in range(1, 7):
            scores[idx, f-1] = counts[f] * f
            if counts[f] > 0: sat_mask[idx, f-1] = 1
        
        # X of a Kind
        for k, p_idx in OF_A_KIND_TO_IDX.items():
            if k == 5: 
                if np.any(counts[1:] >= 5):
                    scores[idx, p_idx] = YATZY_SCORE
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
        u_vals = np.unique(d_arr)
        if np.array_equal(u_vals, [1,2,3,4,5]): 
            scores[idx, 11] = SMALL_STRAIGHT_SCORE
            sat_mask[idx, 11] = 1
        if np.array_equal(u_vals, [2,3,4,5,6]): 
            scores[idx, 12] = LARGE_STRAIGHT_SCORE
            sat_mask[idx, 12] = 1
            
        # Full House
        if np.any(counts == 3) and np.any(counts == 2):
            scores[idx, 13] = (np.where(counts==3)[0][0]*3 + np.where(counts==2)[0][0]*2)
            sat_mask[idx, 13] = 1

        # Chance
        scores[idx, 14] = d_sum
        sat_mask[idx, 14] = 1

        # Pre-calc keep strategy (keep max counts)
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

    This is effectively a base-6 conversion, where each die position represents a
    digit. This allows for instant lookup in the pre-calculated tables.
    Example: (1,1,1,1,1) -> 0, (6,6,6,6,6) -> 7775.

    Args:
        dice: A numpy array of 5 integers representing the dice roll.

    Returns:
        The unique integer index for the roll.
    """
    k = 0
    for i in range(5): 
        k = k * 6 + (dice[i] - 1)
    return k

@njit(nogil=True)
def _reroll(dice: np.ndarray, roll_idx: int, out: np.ndarray, rng):
    """
    Performs a re-roll based on a pre-calculated "greedy" strategy.

    The strategy is to keep the largest group of identical dice with the highest
    face value and re-roll the rest. The decision is looked up from the TBL_KEEP_* tables.

    Args:
        dice: The current 5-dice roll.
        roll_idx: The encoded index of the current roll.
        out: A pre-allocated array to store the new roll result.
        rng: The Numba-compatible random number generator instance.
    """
    cnt = TBL_KEEP_CNT[roll_idx]
    if cnt == 5:
        for i in range(5):
            out[i] = dice[i]
        return
        
    val = TBL_KEEP_VAL[roll_idx]
    for i in range(cnt): 
        out[i] = val
    for i in range(cnt, 5): 
        out[i] = np.random.randint(1, 7)

@njit(nogil=True)
def _play_game_optimized(stats0: np.ndarray, stats1: np.ndarray, stats2: np.ndarray,
                         d0: np.ndarray, d1: np.ndarray, d2: np.ndarray,
                         rng) -> Tuple[int, bool, bool]:
    """
    Simulates one full game of Yatzy.

    This function simulates all 15 rounds of a game. In each round, it performs
    up to three rolls, uses the strategy to make re-roll decisions, and selects the best
    category to score based the a fixed priority list.

    Args:
        stats0, stats1, stats2: Arrays to accumulate category satisfaction counts
                                for each of the three possible rolls in a turn.
        d0, d1, d2: Pre-allocated arrays to hold dice values for the three rolls.
        rng: The Numba-compatible random number generator instance.

    Returns:
        A tuple containing:
        - The final total score for the game.
        - A boolean indicating if the upper section bonus was achieved.
        - A boolean indicating if a Yatzy was scored.
    """
    allowed = ALL_CATEGORIES_MASK
    total = 0
    upper = 0
    got_yatzy = False
    
    for _ in range(NUM_CATEGORIES): 
        # Roll 1
        d0[:] = np.random.randint(1, 7, size=5).astype(np.int8)
        idx0 = _encode(d0)
        
        row0 = TBL_SAT[idx0]
        for c in range(NUM_CATEGORIES): 
            stats0[c] += row0[c]
        
        # Roll 2
        _reroll(d0, idx0, d1, np.random)
        idx1 = _encode(d1)
        row1 = TBL_SAT[idx1]
        for c in range(NUM_CATEGORIES): 
            stats1[c] += row1[c]
        
        # Roll 3
        _reroll(d1, idx1, d2, np.random)
        idx2 = _encode(d2)
        row2 = TBL_SAT[idx2]
        for c in range(NUM_CATEGORIES): 
            stats2[c] += row2[c]
        
        best_sc = -1
        best_id = -1
        scores = TBL_SCORES[idx2]
        
        # Category selection: Iterate through the fixed priority list (TBL_PRIO)
        # and pick the highest-scoring available category.
        for i in range(NUM_CATEGORIES):
            cat = TBL_PRIO[i]
            # Check if the category 'cat' is available using a bitwise AND.
            if (allowed >> cat) & 1:
                s = scores[cat] # Get the score for this category from the lookup table.
                if s > best_sc:
                    best_sc = s
                    best_id = cat
        
        total += best_sc
        if best_id < 6: 
            upper += best_sc
        if best_id == YATZY_IDX and best_sc > 0: 
            got_yatzy = True
        
        # Mark the chosen category as used by flipping its bit in the 'allowed' mask.
        allowed &= ~(1 << best_id)
        
    bonus = UPPER_SECTION_BONUS_SCORE if upper >= UPPER_SECTION_BONUS_THRESHOLD else 0
    return np.int16(total + bonus), bonus > 0, got_yatzy

@njit(nogil=True)
def _simulation_core(count: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The core, Numba-optimized simulation loop.

    This function is designed to be called from a Python-space wrapper that provides
    a pre-instantiated random number generator.

    Args:
        count: The number of games to simulate.
        rng: An initialized Numba-compatible random number generator instance.

    Returns:
        Aggregated results for the simulation batch.
    """
    scores_out = np.empty(count, dtype=np.int16)
    flags_out = np.empty(count, dtype=np.int8)
    l_s0, l_s1, l_s2 = np.zeros((3, NUM_CATEGORIES), dtype=np.int64)
    d0, d1, d2 = np.empty((3, 5), dtype=np.int8)

    for i in range(count):
        sc, bon, ytz = _play_game_optimized(l_s0, l_s1, l_s2, d0, d1, d2, rng)
        scores_out[i] = sc
        flags_out[i] = (1 if bon else 0) | (2 if ytz else 0)
    return scores_out, flags_out, l_s0, l_s1, l_s2

@njit(parallel=True, nogil=True)
def _simulation_core_parallel(count: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The core, Numba-optimized parallel simulation loop.

    This function uses Numba's `prange` to run simulations in parallel, offering
    higher performance than Python-level threading. It aggregates results from all threads.

    Args:
        count: The total number of games to simulate.
        seed: A seed to initialize the thread-local random number generators.

    Returns:
        Aggregated results for the simulation batch.
    """
    np.random.seed(seed)
    scores_out = np.empty(count, dtype=np.int16)
    flags_out = np.empty(count, dtype=np.int8)
    agg_s0, agg_s1, agg_s2 = np.zeros((3, NUM_CATEGORIES), dtype=np.int64)

    for i in prange(count):
        # Thread-local arrays for dice and stats
        d0, d1, d2 = np.empty((3, 5), dtype=np.int8)
        l_s0, l_s1, l_s2 = np.zeros((3, NUM_CATEGORIES), dtype=np.int64)
        sc, bon, ytz = _play_game_optimized(l_s0, l_s1, l_s2, d0, d1, d2, np.random)
        scores_out[i] = sc
        flags_out[i] = (1 if bon else 0) | (2 if ytz else 0)
        agg_s0, agg_s1, agg_s2 = agg_s0 + l_s0, agg_s1 + l_s1, agg_s2 + l_s2
    return scores_out, flags_out, agg_s0, agg_s1, agg_s2

def _worker_sim_batch(count: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a batch of Yatzy games in a single, isolated thread.

    This function is the target for the parallel ThreadPoolExecutor. It initializes its
    own random number generator from a given seed to ensure thread safety and
    reproducibility. It then runs a specified number of games and aggregates their results.

    Args:
        count: The number of games to simulate in this batch.
        seed: A seed for the local random number generator.

    Returns:
        A tuple containing the results for the batch:
        - (np.ndarray): An array of final scores for each game.
        - (np.ndarray): An array of bit flags for each game (bonus, yatzy).
        - (np.ndarray, np.ndarray, np.ndarray): Aggregated category satisfaction counts for rolls 1, 2, and 3.
    """
    rng = np.random.default_rng(seed)
    return _simulation_core(count, rng)

# --- DRIVER LOGIC ---

def run_simulation_parallel(total_count: int, batch_size: int = None, main_rng: np.random.Generator = None, use_numba_parallel: bool = True, result_callback=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Manages the parallel execution of the Yatzy simulation.

    This function orchestrates the simulation by splitting the total number of games
    into smaller batches and distributing them across a ThreadPoolExecutor. It
    dynamically calculates a reasonable batch size if one is not provided.
    It aggregates results from all worker threads and displays real-time progress.

    Args:
        total_count: The total number of games to simulate.
        batch_size: The number of games to simulate per worker batch.
        main_rng: The main random number generator, used to seed the workers.
        use_numba_parallel: If True, use Numba's `parallel=True` model.
        result_callback: An optional function to call with the results of each batch.

    Returns:
        A tuple containing the aggregated results from all simulations:
        - Aggregated category satisfaction counts for rolls 1, 2, and 3.
    """
    local_rng = main_rng if main_rng is not None else np.random.default_rng()
    if batch_size is None:
        cpu_count = os.cpu_count() or 4
        target_chunks = cpu_count * 4
        batch_size = max(1000, total_count // target_chunks)
        batch_size = min(batch_size, 1_000_000)

    agg_s0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_s1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_s2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    
    start_time = time.time()
    
    if use_numba_parallel:
        remaining = total_count
        while remaining > 0:
            count = min(batch_size, remaining)
            seed = local_rng.integers(0, 2**30)
            res_scores, res_flags, r_s0, r_s1, r_s2 = _simulation_core_parallel(count, seed)
            agg_s0, agg_s1, agg_s2 = agg_s0 + r_s0, agg_s1 + r_s1, agg_s2 + r_s2
            remaining -= count
            if result_callback: result_callback(res_scores, res_flags)
            completed = total_count - remaining
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_count - completed) / rate if rate > 0 else 0
            pct = completed / total_count * 100
            print(f"\rSimulating: {pct:5.1f}% | {rate:9,.0f} games/s | ETA: {eta:3.0f}s ", end="")
    else: # Original ThreadPoolExecutor implementation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            remaining = total_count
            while remaining > 0:
                count = min(batch_size, remaining)
                seed = local_rng.integers(0, 2**30)
                futures.append(executor.submit(_worker_sim_batch, count, seed))
                remaining -= count
            
            completed = 0
            for f in concurrent.futures.as_completed(futures):
                res_scores, res_flags, r_s0, r_s1, r_s2 = f.result()
                agg_s0, agg_s1, agg_s2 = agg_s0 + r_s0, agg_s1 + r_s1, agg_s2 + r_s2
                completed += len(res_scores)
                if result_callback: result_callback(res_scores, res_flags)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_count - completed) / rate if rate > 0 else 0
                pct = completed / total_count * 100
                print(f"\rSimulating: {pct:5.1f}% | {rate:9,.0f} games/s | ETA: {eta:3.0f}s ", end="")
            
    print() # Newline after loop
    
    return agg_s0, agg_s1, agg_s2

# --- MAIN EXECUTION ---

def run_suite(args):
    """
    Main driver for the Yatzy simulation suite.

    Parses command-line arguments and executes the requested simulation mode(s),
    either a distribution analysis or a deviation study. Handles file I/O for
    saving results and metadata.
    """
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    main_rng = np.random.default_rng(args.seed)
    
    # Mode 1: Distribution Analysis
    if args.n:
        print(f"\n=== MODE 1: DISTRIBUTION ANALYSIS ({args.n:,} sim) ===")
        start_t = time.time()
        
        # Initialize bins for score distributions. Max score is 375.
        score_bins = np.zeros(376, dtype=np.int64)
        bins_ny_nb = np.zeros(376, dtype=np.int64)
        bins_ny_yb = np.zeros(376, dtype=np.int64)
        bins_yy_nb = np.zeros(376, dtype=np.int64)
        bins_yy_yb = np.zeros(376, dtype=np.int64)
        total_score_sum = 0
        
        # This function will now be called inside the simulation loop for each completed batch
        def process_batch(scores, flags):
            nonlocal total_score_sum
            total_score_sum += np.sum(scores)
            mask_bonus = (flags & 1) > 0
            mask_yatzy = (flags & 2) > 0
            
            score_bins   += np.bincount(scores, minlength=376)
            bins_ny_nb   += np.bincount(scores[~mask_yatzy & ~mask_bonus], minlength=376)
            bins_ny_yb   += np.bincount(scores[~mask_yatzy &  mask_bonus], minlength=376)
            bins_yy_nb   += np.bincount(scores[ mask_yatzy & ~mask_bonus], minlength=376)
            bins_yy_yb   += np.bincount(scores[ mask_yatzy &  mask_bonus], minlength=376)

        # The simulation function now takes a callback to process results incrementally
        s0, s1, s2 = run_simulation_parallel(args.n, main_rng=main_rng, result_callback=process_batch)
        print("\nProcessing complete. Saving data...")
        
        with open(out_dir / f"dist_scores_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Score", "Count_Total", "NoYatzy_NoBonus", "NoYatzy_YesBonus", "YesYatzy_NoBonus", "YesYatzy_YesBonus"])
            for s in range(376):
                if score_bins[s] > 0:
                    w.writerow([s, score_bins[s], bins_ny_nb[s], bins_ny_yb[s], bins_yy_nb[s], bins_yy_yb[s]])

        tot_r = args.n * NUM_CATEGORIES
        with open(out_dir / f"dist_categories_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Roll1_Hits", "Roll1_Prob", "Roll2_Hits", "Roll2_Prob", "Roll3_Hits", "Roll3_Prob"])
            for i, cat in enumerate(CATEGORY_NAMES):
                w.writerow([cat, s0[i], s0[i]/tot_r, s1[i], s1[i]/tot_r, s2[i], s2[i]/tot_r])
                
        elapsed_tot = time.time() - start_t
        meta = {
            "mode": "distribution",
            "count": args.n,
            "seed": args.seed,
            "elapsed_sec": round(elapsed_tot, 2),
            "mean_score": float(total_score_sum / args.n),
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
        results = []
        summary_data = {cat: {step: [] for step in steps} for cat in CATEGORY_NAMES}
        start_t_study = time.time()
        
        for step in steps:
            for r in range(reps):
                print(f"\rRunning simulation: Step {step}, Rep {r+1}/{reps}...", end="")
                _, _, s0, _, _ = run_simulation_parallel(step, batch_size=max(1000, step//(os.cpu_count() or 1)), main_rng=main_rng)
                
                total_rolls = step * NUM_CATEGORIES
                obs_probs = s0 / total_rolls
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

        print("\nAnalysis complete. Saving data...")
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
                    row.append(f"{avg_dev:.4f}")
                w.writerow(row)

        meta_study = {
            "mode": "deviation_study",
            "steps": steps,
            "reps": reps,
            "seed": args.seed,
            "elapsed_sec": time.time() - start_t_study
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
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random number generator for reproducibility.")

    args = parser.parse_args()
    
    if not args.n and not args.study:
        parser.print_help()
    else:
        run_suite(args)