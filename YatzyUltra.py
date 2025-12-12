import argparse
import csv
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from numba import njit, prange, get_num_threads
    from numba import int32 as nb_int32, int8 as nb_int8, boolean as nb_boolean
    NUMBA_AVAILABLE = True
except Exception:  # NumPy/Numba incompatibility or missing package
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore
    prange = range  # type: ignore

    def get_num_threads():
        return 1

    nb_int32 = int  # type: ignore
    nb_int8 = int  # type: ignore
    nb_boolean = bool  # type: ignore


## --- CONFIGURATION ---
debug = False

CATEGORY_NAMES = [
    "Upper1",
    "Upper2",
    "Upper3",
    "Upper4",
    "Upper5",
    "Upper6",
    "OfAKind2",
    "OfAKind3",
    "OfAKind4",
    "OfAKind5",
    "TwoPairs",
    "SmallStraight",
    "LargeStraight",
    "FullHouse",
    "Chance",
]

CATEGORY_INDEX = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
FACE_TO_UPPER_INDEX = {face: face - 1 for face in range(1, 7)}
OF_A_KIND_TO_INDEX = {2: 6, 3: 7, 4: 8, 5: 9}
YATZY_CATEGORY_INDEX = OF_A_KIND_TO_INDEX[5]

SMALL_STRAIGHT_TEMPLATE = np.array([1, 2, 3, 4, 5], dtype=np.int8)
LARGE_STRAIGHT_TEMPLATE = np.array([2, 3, 4, 5, 6], dtype=np.int8)

NUM_CATEGORIES = len(CATEGORY_NAMES)
NUM_ROUNDS = NUM_CATEGORIES

ROLL_STATE_COUNT = 6 ** 5
CATEGORY_PRIORITY = np.array([
    FACE_TO_UPPER_INDEX[6],
    FACE_TO_UPPER_INDEX[5],
    FACE_TO_UPPER_INDEX[4],
    FACE_TO_UPPER_INDEX[3],
    FACE_TO_UPPER_INDEX[2],
    FACE_TO_UPPER_INDEX[1],
    OF_A_KIND_TO_INDEX[5],
    OF_A_KIND_TO_INDEX[4],
    OF_A_KIND_TO_INDEX[3],
    OF_A_KIND_TO_INDEX[2],
    CATEGORY_INDEX["TwoPairs"],
    CATEGORY_INDEX["LargeStraight"],
    CATEGORY_INDEX["SmallStraight"],
    CATEGORY_INDEX["FullHouse"],
    CATEGORY_INDEX["Chance"],
], dtype=np.int8)



## --- HELPERS ---

def _format_duration(seconds):
    seconds_int = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_elapsed(seconds):
    seconds_float = max(0.0, float(seconds))
    if seconds_float < 0.001:
        return "0ms"
    if seconds_float < 1.0:
        return f"{seconds_float * 1000:.0f}ms"
    if seconds_float < 60.0:
        return f"{seconds_float:.2f}s"
    return _format_duration(seconds_float)


def _format_completion_timestamp(seconds_from_now):
    if not math.isfinite(seconds_from_now) or seconds_from_now < 0:
        return None
    eta_time = datetime.now().astimezone() + timedelta(seconds=seconds_from_now)
    return eta_time.isoformat(timespec="seconds")


def RollDice(count, rng=None):
    """Roll `count` dice. Returns a NumPy array when an RNG is provided, otherwise a list."""
    generator = rng if rng is not None else np.random.default_rng()
    dice = generator.integers(1, 7, size=count, dtype=np.int8)
    return dice if rng is not None else dice.tolist()


def _encode_roll(dice_array):
    key = 0
    for value in dice_array:
        key = key * 6 + (int(value) - 1)
    return key


def _reroll_keep_most_common_array(dice_array, rng, encoded_roll=None, out=None):
    roll_index = _encode_roll(dice_array) if encoded_roll is None else encoded_roll
    keep_count = int(ROLL_KEEP_COUNT[roll_index])

    if keep_count == 5:
        if out is None:
            return dice_array.copy()
        np.copyto(out, dice_array)
        return out

    result = np.empty(5, dtype=np.int8) if out is None else out
    keep_value = int(ROLL_KEEP_VALUE[roll_index])
    if keep_count:
        result[:keep_count] = keep_value
    if keep_count < 5:
        generator = rng if rng is not None else np.random.default_rng()
        rerolls = generator.integers(1, 7, size=5 - keep_count, dtype=np.int8)
        result[keep_count:] = rerolls
    return result


def _cleanup_pycache(root_directory):
    """Remove __pycache__ directories under root_directory."""
    for dirpath, dirnames, _ in os.walk(root_directory):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path, ignore_errors=True)


def RerollKeepMostCommon(dice, rng=None):
    """Compatibility wrapper that accepts lists or arrays."""
    dice_array = np.asarray(dice, dtype=np.int8)
    generator = rng if rng is not None else np.random.default_rng()
    result = _reroll_keep_most_common_array(dice_array, generator)
    return result if rng is not None else result.tolist()


def _analyze_dice(dice_array):
    counts = np.bincount(dice_array, minlength=7)
    sorted_dice = np.sort(dice_array)
    dice_sum = int(dice_array.sum())
    return counts, sorted_dice, dice_sum


def _score_upper(counts, face):
    return int(counts[face]) * face


def _score_two_pairs(dice_array, counts):
    seen = set()
    pairs = []
    for value in dice_array:
        face = int(value)
        if face in seen:
            continue
        seen.add(face)
        cnt = int(counts[face])
        if cnt == 2 or cnt == 3:
            pairs.append(face)
    if len(pairs) < 2:
        return 0
    return sum(face * 2 for face in pairs)


def _score_n_of_a_kind(counts, required):
    if required == 5:
        return 50 if np.any(counts[1:] >= 5) else 0
    for face in range(6, 0, -1):
        if counts[face] >= required:
            return face * required
    return 0


def _score_straight(sorted_dice, size):
    template = LARGE_STRAIGHT_TEMPLATE if size == 1 else SMALL_STRAIGHT_TEMPLATE
    return 15 + size * 5 if np.array_equal(sorted_dice, template) else 0


def _score_full_house(counts):
    score = 0
    has_pair = False
    has_three = False
    for face in range(1, 7):
        cnt = int(counts[face])
        if cnt == 2:
            has_pair = True
            score += face * 2
        elif cnt == 3:
            has_three = True
            score += face * 3
    return score if has_pair and has_three else 0


def _resolve_keep_strategy(counts):
    counts_without_zero = counts[1:]
    max_count = int(counts_without_zero.max())
    keep_value = 1
    for face in range(6, 0, -1):
        if counts[face] == max_count:
            keep_value = face
            break
    return keep_value, max_count


def EvaluateBonus(points):
    bonus = 50 if points >= 63 else 0
    if debug:
        print("Bonus | " + str(bonus))
    return bonus


def _satisfied_categories(dice_array, counts, sorted_dice, dice_sum):
    satisfied = []

    for face in range(6, 0, -1):
        idx = FACE_TO_UPPER_INDEX[face]
        if _score_upper(counts, face) > 0:
            satisfied.append(idx)

    for required in range(5, 1, -1):
        idx = OF_A_KIND_TO_INDEX[required]
        if _score_n_of_a_kind(counts, required) > 0:
            satisfied.append(idx)

    if _score_two_pairs(dice_array, counts) > 0:
        satisfied.append(CATEGORY_INDEX["TwoPairs"])

    if _score_straight(sorted_dice, 1) > 0:
        satisfied.append(CATEGORY_INDEX["LargeStraight"])
    if _score_straight(sorted_dice, 0) > 0:
        satisfied.append(CATEGORY_INDEX["SmallStraight"])

    if _score_full_house(counts) > 0:
        satisfied.append(CATEGORY_INDEX["FullHouse"])

    if dice_sum > 0:
        satisfied.append(CATEGORY_INDEX["Chance"])

    return satisfied


def _build_roll_cache():
    score_table = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int8)
    sat_matrix = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    sat_lists = [()] * ROLL_STATE_COUNT
    keep_values = np.empty(ROLL_STATE_COUNT, dtype=np.int8)
    keep_counts = np.empty(ROLL_STATE_COUNT, dtype=np.uint8)

    for idx, dice in enumerate(product(range(1, 7), repeat=5)):
        dice_array = np.array(dice, dtype=np.int8)
        counts, sorted_dice, dice_sum = _analyze_dice(dice_array)
        satisfied = _satisfied_categories(dice_array, counts, sorted_dice, dice_sum)
        if satisfied:
            sat_matrix[idx, satisfied] = 1
        sat_lists[idx] = tuple(satisfied)

        keep_value, keep_count = _resolve_keep_strategy(counts)
        keep_values[idx] = keep_value
        keep_counts[idx] = keep_count

        row = score_table[idx]
        for face in range(1, 7):
            row[FACE_TO_UPPER_INDEX[face]] = _score_upper(counts, face)
        for required, category_idx in OF_A_KIND_TO_INDEX.items():
            row[category_idx] = _score_n_of_a_kind(counts, required)
        row[CATEGORY_INDEX["TwoPairs"]] = _score_two_pairs(dice_array, counts)
        row[CATEGORY_INDEX["LargeStraight"]] = _score_straight(sorted_dice, 1)
        row[CATEGORY_INDEX["SmallStraight"]] = _score_straight(sorted_dice, 0)
        row[CATEGORY_INDEX["FullHouse"]] = _score_full_house(counts)
        row[CATEGORY_INDEX["Chance"]] = dice_sum

    return score_table, sat_matrix, tuple(sat_lists), keep_values, keep_counts


ROLL_SCORE_TABLE, ROLL_SAT_MATRIX, ROLL_SAT_LISTS, ROLL_KEEP_VALUE, ROLL_KEEP_COUNT = _build_roll_cache()


def _evaluate_best_category_from_index(roll_index, allowed_mask):
    scores = ROLL_SCORE_TABLE[roll_index]
    best_score = -1
    best_idx = -1

    for idx in CATEGORY_PRIORITY:
        if not allowed_mask[idx]:
            continue
        score = int(scores[idx])
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_score, best_idx


if NUMBA_AVAILABLE:
    CATEGORY_PRIORITY_NUMBA = CATEGORY_PRIORITY.astype(np.int8)

    @njit(cache=True, nogil=True)
    def _encode_roll_numba(dice_array):
        key = 0
        for value in dice_array:
            key = key * 6 + (int(value) - 1)
        return key

    @njit(cache=True, nogil=True)
    def _reroll_keep_most_common_array_numba(dice_array, roll_index, out):
        keep_count = int(ROLL_KEEP_COUNT[roll_index])
        if keep_count == 5:
            for i in range(5):
                out[i] = dice_array[i]
            return

        keep_value = int(ROLL_KEEP_VALUE[roll_index])
        for i in range(keep_count):
            out[i] = keep_value
        for i in range(keep_count, 5):
            out[i] = np.int8(np.random.randint(1, 7))

    @njit(cache=True, nogil=True)
    def _evaluate_best_category_from_index_numba(roll_index, allowed_bits):
        scores = ROLL_SCORE_TABLE[roll_index]
        best_score = -1
        best_idx = -1
        for j in range(CATEGORY_PRIORITY_NUMBA.shape[0]):
            idx = int(CATEGORY_PRIORITY_NUMBA[j])
            if (allowed_bits >> idx) & 1 == 0:
                continue
            score = int(scores[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_score, best_idx

    @njit(cache=True, nogil=True)
    def _play_yatzy_numba(dice0, dice1, dice2):
        allowed_bits = (1 << NUM_CATEGORIES) - 1
        total_points = 0
        upper_points = 0

        for _ in range(NUM_ROUNDS):
            for j in range(5):
                dice0[j] = np.int8(np.random.randint(1, 7))
            idx0 = _encode_roll_numba(dice0)

            _reroll_keep_most_common_array_numba(dice0, idx0, dice1)
            idx1 = _encode_roll_numba(dice1)

            _reroll_keep_most_common_array_numba(dice1, idx1, dice2)
            idx2 = _encode_roll_numba(dice2)

            score, chosen_idx = _evaluate_best_category_from_index_numba(idx2, allowed_bits)
            total_points += score
            if chosen_idx < 6:
                upper_points += score
            allowed_bits &= ~(1 << chosen_idx)

        bonus = 50 if upper_points >= 63 else 0
        total_points += bonus
        return total_points

    @njit(parallel=True, cache=True, nogil=True)
    def _simulate_chunk_numba(count, seed):
        np.random.seed(seed)
        results = np.empty(count, dtype=np.uint16)

        for i in prange(count):
            dice0 = np.empty(5, dtype=np.int8)
            dice1 = np.empty(5, dtype=np.int8)
            dice2 = np.empty(5, dtype=np.int8)

            results[i] = np.uint16(_play_yatzy_numba(dice0, dice1, dice2))

        return results


## --- MAIN GAMEPLAY ---

def PlayYatzy(rng=None, stats0=None, stats1=None, stats2=None, allowed_mask=None):
    generator = rng if rng is not None else np.random.default_rng()

    if stats0 is None:
        stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    else:
        stats0.fill(0)
    if stats1 is None:
        stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    else:
        stats1.fill(0)
    if stats2 is None:
        stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    else:
        stats2.fill(0)

    if allowed_mask is None:
        allowed_mask = np.ones(NUM_CATEGORIES, dtype=bool)
    else:
        allowed_mask.fill(True)

    roll_sat_matrix = ROLL_SAT_MATRIX
    roll_sat_lists = ROLL_SAT_LISTS

    dice0 = np.empty(5, dtype=np.int8)
    dice1 = np.empty(5, dtype=np.int8)
    dice2 = np.empty(5, dtype=np.int8)

    total_points = 0
    upper_points = 0
    got_yatzy = False

    for round_idx in range(NUM_ROUNDS):
        dice0[:] = generator.integers(1, 7, size=5, dtype=np.int8)
        idx0 = _encode_roll(dice0)
        stats0 += roll_sat_matrix[idx0]

        _reroll_keep_most_common_array(dice0, generator, idx0, out=dice1)
        idx1 = _encode_roll(dice1)
        stats1 += roll_sat_matrix[idx1]

        _reroll_keep_most_common_array(dice1, generator, idx1, out=dice2)
        idx2 = _encode_roll(dice2)
        stats2 += roll_sat_matrix[idx2]

        score, chosen_idx = _evaluate_best_category_from_index(idx2, allowed_mask)
        total_points += score
        if chosen_idx < 6:
            upper_points += score
        if chosen_idx == YATZY_CATEGORY_INDEX and score > 0:
            got_yatzy = True
        allowed_mask[chosen_idx] = False

        if debug:
            round_no = round_idx + 1
            satisfied0 = [CATEGORY_NAMES[i] for i in roll_sat_lists[idx0]]
            satisfied1 = [CATEGORY_NAMES[i] for i in roll_sat_lists[idx1]]
            satisfied2 = [CATEGORY_NAMES[i] for i in roll_sat_lists[idx2]]
            print(f"Satisfied 0 (Round {round_no}) | {satisfied0}")
            print(f"Satisfied 1 (Round {round_no}) | {satisfied1}")
            print(f"Satisfied 2 (Round {round_no}) | {satisfied2}")
            print(f"Round {round_no} | {CATEGORY_NAMES[chosen_idx]} | {score}")

    bonus = EvaluateBonus(upper_points)
    total_points += bonus

    return total_points, bonus > 0, got_yatzy


def _simulate_numba_backend(count):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    seed_sequence = np.random.SeedSequence()
    seed = int(seed_sequence.generate_state(1)[0])

    start = time.time()
    results = _simulate_chunk_numba(count, seed)
    elapsed_ms = (time.time() - start) * 1000

    return results, elapsed_ms, get_num_threads()




def _update_score_counts(counts, results):
    results = np.asarray(results, dtype=np.int32)
    if results.size == 0:
        return counts
    max_score = int(results.max())
    if max_score >= counts.size:
        expanded = np.zeros(max_score + 1, dtype=np.int64)
        expanded[: counts.size] = counts
        counts = expanded
    chunk_counts = np.bincount(results, minlength=max_score + 1)
    counts[: chunk_counts.size] += chunk_counts
    return counts


@dataclass(frozen=True)
class SimulationSummary:
    count: int
    elapsed_ms: float
    mean: float
    std: float
    min_score: int
    max_score: int
    counts: np.ndarray


def _summarize_counts(counts, elapsed_ms):
    counts = np.asarray(counts, dtype=np.int64)
    total_games = int(counts.sum())
    if total_games == 0:
        return SimulationSummary(
            count=0,
            elapsed_ms=elapsed_ms,
            mean=0.0,
            std=0.0,
            min_score=0,
            max_score=0,
            counts=counts.copy(),
        )

    nonzero_scores = np.nonzero(counts)[0]
    min_score = int(nonzero_scores[0])
    max_score = int(nonzero_scores[-1])

    frequencies = counts[nonzero_scores].astype(np.float64, copy=False)
    scores = nonzero_scores.astype(np.float64, copy=False)
    mean_val = float(np.dot(scores, frequencies) / total_games)
    squared_mean = float(np.dot(scores * scores, frequencies) / total_games)
    variance = max(0.0, squared_mean - mean_val * mean_val)
    std_val = math.sqrt(variance)

    return SimulationSummary(
        count=total_games,
        elapsed_ms=elapsed_ms,
        mean=mean_val,
        std=std_val,
        min_score=min_score,
        max_score=max_score,
        counts=counts.copy(),
    )


def _format_eta_message(remaining, overall_rate):
    if remaining <= 0:
        return "complete"
    if overall_rate is None or not math.isfinite(overall_rate) or overall_rate <= 0:
        return "estimating..."
    eta_seconds = remaining / overall_rate
    completion_str = _format_completion_timestamp(eta_seconds)
    duration_str = _format_duration(eta_seconds)
    return f"{duration_str} (finish ~ {completion_str})" if completion_str else duration_str


def _format_progress_message(processed, count, batch_size, chunk_elapsed, start_time):
    percent_complete = processed / count * 100.0
    chunk_time_str = _format_elapsed(chunk_elapsed)
    chunk_rate = batch_size / chunk_elapsed if chunk_elapsed > 0 else None
    chunk_rate_str = (
        f"{chunk_rate:,.0f} games/s" if chunk_rate is not None and math.isfinite(chunk_rate) else "n/a"
    )
    total_elapsed = time.time() - start_time
    overall_rate = processed / total_elapsed if total_elapsed > 0 else None
    overall_rate_str = (
        f"{overall_rate:,.0f} games/s" if overall_rate is not None and math.isfinite(overall_rate) else "n/a"
    )
    eta_message = _format_eta_message(count - processed, overall_rate)
    return (
        f"Chunk complete: {processed:,}/{count:,} ({percent_complete:.2f}%) | "
        f"chunk {batch_size:,} in {chunk_time_str} ({chunk_rate_str}) | "
        f"overall {overall_rate_str} | ETA {eta_message}"
    )


def _print_summary(summary: SimulationSummary, threads_used: Optional[int]):
    elapsed_seconds = summary.elapsed_ms / 1000.0
    throughput = summary.count / elapsed_seconds if elapsed_seconds > 0 else float("inf")

    print()
    print("--- SCORE DISTRIBUTION SUMMARY ---")
    print(f"Games simulated: {summary.count:,}")
    print(f"Elapsed time: {_format_elapsed(elapsed_seconds)}")
    if math.isfinite(throughput):
        print(f"Throughput: {throughput:,.0f} games/s")
    else:
        print("Throughput: n/a")
    if threads_used is not None:
        print(f"Numba threads used: {threads_used}")
    print(f"Mean score: {summary.mean:.3f}")
    print(f"Std dev: {summary.std:.3f}")
    print(f"Score range: {summary.min_score} - {summary.max_score}")

    nonzero_scores = np.nonzero(summary.counts)[0]
    if nonzero_scores.size:
        top_indices = nonzero_scores[np.argsort(summary.counts[nonzero_scores])[-5:]]
        top_indices = top_indices[::-1]
        print("Top 5 scores by frequency:")
        for score in top_indices:
            freq = int(summary.counts[score])
            pct = freq / summary.count * 100.0 if summary.count else 0.0
            print(f"  {score:3d}: {freq:,} ({pct:.4f}%)")
    print()


def _save_distribution(output_dir: Union[str, Path], counts: np.ndarray, run_start_unix: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    distribution_path = output_path / f"yatzy_distribution_{run_start_unix}.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["score", "count"])
        for score, count in enumerate(counts):
            if count:
                writer.writerow([score, int(count)])
    return distribution_path


def SimulateRounds(
    count,
    chunk_size=None,
    output_dir="results",
    save_distribution=True,
):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for simulation but is not available.")

    count_int = int(count)
    if count_int <= 0:
        raise ValueError("count must be a positive integer")

    if chunk_size is None:
        chunk_size_value = min(count_int, 10_000_000)
    else:
        chunk_size_value = int(chunk_size)
    if chunk_size_value <= 0:
        raise ValueError("chunk_size must be a positive integer")

    chunk_size_value = max(1, min(chunk_size_value, count_int))

    counts = np.zeros(1, dtype=np.int64)
    processed = 0
    start_time = time.time()
    run_start_unix = int(start_time)
    threads_used: Optional[int] = None

    while processed < count_int:
        batch_size = min(chunk_size_value, count_int - processed)
        chunk_start = time.time()
        results, _, units = _simulate_numba_backend(batch_size)
        chunk_elapsed = time.time() - chunk_start

        counts = _update_score_counts(counts, results)
        processed += batch_size
        if threads_used is None:
            threads_used = units

        progress_message = _format_progress_message(
            processed,
            count_int,
            batch_size,
            chunk_elapsed,
            start_time,
        )
        print(progress_message, flush=True)

        del results

    elapsed_ms = (time.time() - start_time) * 1000.0
    summary = _summarize_counts(counts, elapsed_ms)
    _print_summary(summary, threads_used)

    distribution_path = None
    if save_distribution:
        distribution_path = _save_distribution(output_dir, summary.counts, run_start_unix)
        print(f"Score distribution saved to: {distribution_path}")

    return summary, distribution_path


def _parse_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}")
    return parsed


def _parse_optional_positive_int(value):
    if value.lower() in {"auto", "none"}:
        return None
    return _parse_positive_int(value)

def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Simulate Yatzy games and record the score distribution."
    )
    parser.add_argument(
        "--count",
        type=_parse_positive_int,
        default=1_000_000,
        help="Number of games to simulate (default: 1,000,000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=_parse_optional_positive_int,
        default=None,
        help="Batch size per iteration (default: auto). Use 'auto' to pick a default.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for the score distribution CSV (default: results).",
    )
    parser.add_argument(
        "--no-save-distribution",
        dest="save_distribution",
        action="store_false",
        default=True,
        help="Skip writing the score distribution CSV.",
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action="store_true",
        default=True,
        help="Remove __pycache__ directories after the run (default: enabled).",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Leave __pycache__ directories untouched after the run.",
    )
    return parser


if __name__ == "__main__":
    cli_parser = _build_argument_parser()
    cli_args = cli_parser.parse_args()
    try:
        SimulateRounds(
            count=cli_args.count,
            chunk_size=cli_args.chunk_size,
            output_dir=cli_args.output_dir,
            save_distribution=cli_args.save_distribution,
        )
    finally:
        if cli_args.cleanup:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            _cleanup_pycache(script_directory)
