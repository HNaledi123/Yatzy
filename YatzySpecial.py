import argparse
import csv
import json
import math
import os
import platform
import shutil
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

if NUMBA_AVAILABLE:
    try:
        from numba import cuda
        from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
        CUDA_AVAILABLE = cuda.is_available()
    except Exception:  # CUDA toolchain missing or driver unavailable
        cuda = None  # type: ignore
        create_xoroshiro128p_states = None  # type: ignore
        xoroshiro128p_uniform_float32 = None  # type: ignore
        CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = False
    cuda = None  # type: ignore
    create_xoroshiro128p_states = None  # type: ignore
    xoroshiro128p_uniform_float32 = None  # type: ignore

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
    score_table = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
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
    CATEGORY_PRIORITY_NUMBA = CATEGORY_PRIORITY.astype(np.int16)

    @njit(cache=True)
    def _encode_roll_numba(dice_array):
        key = 0
        for value in dice_array:
            key = key * 6 + (int(value) - 1)
        return key

    @njit(cache=True)
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

    @njit(cache=True)
    def _evaluate_best_category_from_index_numba(roll_index, allowed_mask):
        scores = ROLL_SCORE_TABLE[roll_index]
        best_score = -1
        best_idx = -1
        for j in range(CATEGORY_PRIORITY_NUMBA.shape[0]):
            idx = int(CATEGORY_PRIORITY_NUMBA[j])
            if not allowed_mask[idx]:
                continue
            score = int(scores[idx])
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_score, best_idx

    @njit(cache=True)
    def _play_yatzy_numba(stats0, stats1, stats2, allowed_mask, dice0, dice1, dice2):
        for c in range(NUM_CATEGORIES):
            stats0[c] = 0
            stats1[c] = 0
            stats2[c] = 0
            allowed_mask[c] = True

        total_points = 0
        upper_points = 0
        got_yatzy = False

        for _ in range(NUM_ROUNDS):
            for j in range(5):
                dice0[j] = np.int8(np.random.randint(1, 7))
            idx0 = _encode_roll_numba(dice0)
            row0 = ROLL_SAT_MATRIX[idx0]
            for c in range(NUM_CATEGORIES):
                stats0[c] += int(row0[c])

            _reroll_keep_most_common_array_numba(dice0, idx0, dice1)
            idx1 = _encode_roll_numba(dice1)
            row1 = ROLL_SAT_MATRIX[idx1]
            for c in range(NUM_CATEGORIES):
                stats1[c] += int(row1[c])

            _reroll_keep_most_common_array_numba(dice1, idx1, dice2)
            idx2 = _encode_roll_numba(dice2)
            row2 = ROLL_SAT_MATRIX[idx2]
            for c in range(NUM_CATEGORIES):
                stats2[c] += int(row2[c])

            score, chosen_idx = _evaluate_best_category_from_index_numba(idx2, allowed_mask)
            total_points += score
            if chosen_idx < 6:
                upper_points += score
            if chosen_idx == YATZY_CATEGORY_INDEX and score > 0:
                got_yatzy = True
            allowed_mask[chosen_idx] = False

        bonus = 50 if upper_points >= 63 else 0
        total_points += bonus
        return total_points, bonus > 0, got_yatzy

    @njit(parallel=True, cache=True)
    def _simulate_chunk_numba(count, seed):
        np.random.seed(seed)
        results = np.empty(count, dtype=np.int32)
        bonus_flags = np.empty(count, dtype=np.int8)
        yatzy_hits = np.empty(count, dtype=np.int8)
        stats0_all = np.empty((count, NUM_CATEGORIES), dtype=np.int32)
        stats1_all = np.empty((count, NUM_CATEGORIES), dtype=np.int32)
        stats2_all = np.empty((count, NUM_CATEGORIES), dtype=np.int32)

        for i in prange(count):
            stats0_local = np.empty(NUM_CATEGORIES, dtype=np.int32)
            stats1_local = np.empty(NUM_CATEGORIES, dtype=np.int32)
            stats2_local = np.empty(NUM_CATEGORIES, dtype=np.int32)
            allowed_mask_local = np.empty(NUM_CATEGORIES, dtype=np.bool_)
            dice0 = np.empty(5, dtype=np.int8)
            dice1 = np.empty(5, dtype=np.int8)
            dice2 = np.empty(5, dtype=np.int8)

            total_points, got_bonus, got_yatzy = _play_yatzy_numba(
                stats0_local, stats1_local, stats2_local, allowed_mask_local, dice0, dice1, dice2
            )
            results[i] = total_points
            bonus_flags[i] = np.int8(1 if got_bonus else 0)
            yatzy_hits[i] = np.int8(1 if got_yatzy else 0)

            for c in range(NUM_CATEGORIES):
                stats0_all[i, c] = stats0_local[c]
                stats1_all[i, c] = stats1_local[c]
                stats2_all[i, c] = stats2_local[c]

        return results, bonus_flags, stats0_all, stats1_all, stats2_all, yatzy_hits


if CUDA_AVAILABLE:
    CATEGORY_PRIORITY_CUDA = CATEGORY_PRIORITY.astype(np.int16)

    @cuda.jit(device=True)
    def _encode_roll_cuda(dice_array):
        key = 0
        for i in range(5):
            key = key * 6 + (int(dice_array[i]) - 1)
        return key

    @cuda.jit(device=True)
    def _roll_die_cuda(rng_states, thread_id):
        val = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * 6.0)
        if val >= 6:
            val = 5
        return val + 1

    @cuda.jit
    def _simulate_kernel_cuda(rng_states, results, bonus_hits, yatzy_hits, stats0_all, stats1_all, stats2_all,
                              roll_score_table, roll_sat_matrix, roll_keep_value, roll_keep_count, category_priority,
                              yatzy_category_index):
        idx = cuda.grid(1)
        if idx >= results.shape[0]:
            return

        stats0_local = cuda.local.array(NUM_CATEGORIES, nb_int32)
        stats1_local = cuda.local.array(NUM_CATEGORIES, nb_int32)
        stats2_local = cuda.local.array(NUM_CATEGORIES, nb_int32)
        allowed_mask_local = cuda.local.array(NUM_CATEGORIES, nb_boolean)
        dice0 = cuda.local.array(5, nb_int8)
        dice1 = cuda.local.array(5, nb_int8)
        dice2 = cuda.local.array(5, nb_int8)

        for c in range(NUM_CATEGORIES):
            stats0_local[c] = 0
            stats1_local[c] = 0
            stats2_local[c] = 0
            allowed_mask_local[c] = True

        total_points = 0
        upper_points = 0
        got_yatzy = False

        for _ in range(NUM_ROUNDS):
            for j in range(5):
                dice0[j] = _roll_die_cuda(rng_states, idx)
            idx0 = _encode_roll_cuda(dice0)
            row0 = roll_sat_matrix[idx0]
            for c in range(NUM_CATEGORIES):
                stats0_local[c] += int(row0[c])

            keep_count0 = int(roll_keep_count[idx0])
            keep_value0 = int(roll_keep_value[idx0])
            for j in range(keep_count0):
                dice1[j] = keep_value0
            for j in range(keep_count0, 5):
                dice1[j] = _roll_die_cuda(rng_states, idx)
            idx1 = _encode_roll_cuda(dice1)
            row1 = roll_sat_matrix[idx1]
            for c in range(NUM_CATEGORIES):
                stats1_local[c] += int(row1[c])

            keep_count1 = int(roll_keep_count[idx1])
            keep_value1 = int(roll_keep_value[idx1])
            for j in range(keep_count1):
                dice2[j] = keep_value1
            for j in range(keep_count1, 5):
                dice2[j] = _roll_die_cuda(rng_states, idx)
            idx2 = _encode_roll_cuda(dice2)
            row2 = roll_sat_matrix[idx2]
            for c in range(NUM_CATEGORIES):
                stats2_local[c] += int(row2[c])

            best_score = -1
            best_idx = 0
            for j in range(NUM_CATEGORIES):
                cat_idx = int(category_priority[j])
                if not allowed_mask_local[cat_idx]:
                    continue
                score = int(roll_score_table[idx2, cat_idx])
                if score > best_score:
                    best_score = score
                    best_idx = cat_idx

            total_points += best_score
            if best_idx < 6:
                upper_points += best_score
            if best_idx == yatzy_category_index and best_score > 0:
                got_yatzy = True
            allowed_mask_local[best_idx] = False

        bonus = 50 if upper_points >= 63 else 0
        total_points += bonus

        results[idx] = total_points
        bonus_hits[idx] = 1 if bonus > 0 else 0
        yatzy_hits[idx] = 1 if got_yatzy else 0
        for c in range(NUM_CATEGORIES):
            stats0_all[idx, c] = stats0_local[c]
            stats1_all[idx, c] = stats1_local[c]
            stats2_all[idx, c] = stats2_local[c]


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


def _simulate_chunk(task):
    count, seed = task
    rng = np.random.default_rng(seed)

    results = np.empty(count, dtype=np.int32)
    bonus_hits = 0
    yatzy_flags = np.empty(count, dtype=np.uint8)
    bonus_flags = np.empty(count, dtype=np.uint8)
    agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    allowed_mask = np.empty(NUM_CATEGORIES, dtype=bool)

    for i in range(count):
        points, got_bonus, got_yatzy = PlayYatzy(rng, tmp_stats0, tmp_stats1, tmp_stats2, allowed_mask)
        results[i] = points
        if got_bonus:
            bonus_hits += 1
        yatzy_flags[i] = 1 if got_yatzy else 0
        bonus_flags[i] = 1 if got_bonus else 0
        agg_stats0 += tmp_stats0
        agg_stats1 += tmp_stats1
        agg_stats2 += tmp_stats2

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2, yatzy_flags, bonus_flags


def _simulate_python_backend(count, processes):
    available_cpus = os.cpu_count() or 1
    if processes is None:
        processes = min(available_cpus, count)
    else:
        processes = max(1, min(processes, available_cpus, count))

    base, extra = divmod(count, processes)
    chunk_sizes = [base + (1 if i < extra else 0) for i in range(processes)]
    chunk_sizes = [size for size in chunk_sizes if size > 0]

    seed_sequence = np.random.SeedSequence()
    child_seeds = seed_sequence.spawn(len(chunk_sizes))
    tasks = [(size, int(seed.generate_state(1)[0])) for size, seed in zip(chunk_sizes, child_seeds)]

    start = time.time()
    if len(tasks) == 1:
        chunk_results = [_simulate_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            chunk_results = list(pool.map(_simulate_chunk, tasks))
    elapsed_ms = (time.time() - start) * 1000

    agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    results = np.empty(count, dtype=np.int32)
    yatzy_flags = np.empty(count, dtype=np.uint8)
    bonus_flags = np.empty(count, dtype=np.uint8)
    bonus_hits = 0
    offset = 0
    for (scores, chunk_bonus, s0, s1, s2, chunk_yatzy, chunk_bonus_flags), size in zip(chunk_results, chunk_sizes):
        end = offset + size
        results[offset:end] = scores
        yatzy_flags[offset:end] = chunk_yatzy
        bonus_flags[offset:end] = chunk_bonus_flags
        offset = end
        bonus_hits += chunk_bonus
        agg_stats0 += s0
        agg_stats1 += s1
        agg_stats2 += s2

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2, yatzy_flags, bonus_flags, elapsed_ms, processes


def _simulate_numba_backend(count):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    seed_sequence = np.random.SeedSequence()
    seed = int(seed_sequence.generate_state(1)[0])

    start = time.time()
    results, bonus_flags_array, stats0_all, stats1_all, stats2_all, yatzy_hits_array = _simulate_chunk_numba(count, seed)
    elapsed_ms = (time.time() - start) * 1000

    agg_stats0 = stats0_all.sum(axis=0, dtype=np.int64)
    agg_stats1 = stats1_all.sum(axis=0, dtype=np.int64)
    agg_stats2 = stats2_all.sum(axis=0, dtype=np.int64)
    bonus_hits = int(bonus_flags_array.sum())

    return (
        results,
        bonus_hits,
        agg_stats0,
        agg_stats1,
        agg_stats2,
        yatzy_hits_array.astype(np.uint8),
        bonus_flags_array.astype(np.uint8),
        elapsed_ms,
        get_num_threads(),
    )


def _simulate_cuda_backend(count, threads_per_block=128):
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend requested but CUDA is not available.")

    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be a positive integer")

    blocks_per_grid = (count + threads_per_block - 1) // threads_per_block
    total_threads = blocks_per_grid * threads_per_block

    roll_score_table_dev = cuda.to_device(ROLL_SCORE_TABLE.astype(np.int16))
    roll_sat_matrix_dev = cuda.to_device(ROLL_SAT_MATRIX.astype(np.uint8))
    roll_keep_value_dev = cuda.to_device(ROLL_KEEP_VALUE.astype(np.int8))
    roll_keep_count_dev = cuda.to_device(ROLL_KEEP_COUNT.astype(np.uint8))
    category_priority_dev = cuda.to_device(CATEGORY_PRIORITY_CUDA)

    results_dev = cuda.device_array(count, dtype=np.int32)
    bonus_hits_dev = cuda.device_array(count, dtype=np.uint8)
    yatzy_hits_dev = cuda.device_array(count, dtype=np.uint8)
    stats0_dev = cuda.device_array((count, NUM_CATEGORIES), dtype=np.int32)
    stats1_dev = cuda.device_array((count, NUM_CATEGORIES), dtype=np.int32)
    stats2_dev = cuda.device_array((count, NUM_CATEGORIES), dtype=np.int32)

    seed_sequence = np.random.SeedSequence()
    seed = int(seed_sequence.generate_state(1)[0])
    rng_states = create_xoroshiro128p_states(total_threads, seed=seed)

    start = time.time()
    _simulate_kernel_cuda[blocks_per_grid, threads_per_block](
        rng_states,
        results_dev,
        bonus_hits_dev,
        yatzy_hits_dev,
        stats0_dev,
        stats1_dev,
        stats2_dev,
        roll_score_table_dev,
        roll_sat_matrix_dev,
        roll_keep_value_dev,
        roll_keep_count_dev,
        category_priority_dev,
        np.int16(YATZY_CATEGORY_INDEX),
    )
    cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000

    results = results_dev.copy_to_host()
    bonus_flags_array = bonus_hits_dev.copy_to_host()
    yatzy_hits_array = yatzy_hits_dev.copy_to_host()
    stats0_all = stats0_dev.copy_to_host()
    stats1_all = stats1_dev.copy_to_host()
    stats2_all = stats2_dev.copy_to_host()

    agg_stats0 = stats0_all.sum(axis=0, dtype=np.int64)
    agg_stats1 = stats1_all.sum(axis=0, dtype=np.int64)
    agg_stats2 = stats2_all.sum(axis=0, dtype=np.int64)
    bonus_hits = int(bonus_flags_array.sum())

    return (
        results,
        bonus_hits,
        agg_stats0,
        agg_stats1,
        agg_stats2,
        yatzy_hits_array,
        bonus_flags_array,
        elapsed_ms,
        total_threads,
    )


class _SimulationAccumulator:
    """Aggregate simulation statistics incrementally to avoid storing per-game data."""

    def __init__(self, store_results):
        self.store_results = store_results
        self._result_chunks = [] if store_results else None
        self._yatzy_chunks = [] if store_results else None
        self._bonus_chunks = [] if store_results else None
        self.total_count = 0
        self.total_sum = 0.0
        self.total_sum_sq = 0.0
        self.total_bonus_hits = 0
        self.total_yatzy_hits = 0
        self.total_yatzy_and_bonus = 0
        self.total_yatzy_no_bonus = 0
        self.total_no_yatzy_bonus = 0
        self.total_no_yatzy_no_bonus = 0
        self.agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self.agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self.agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self._hist_counts = np.zeros(1, dtype=np.int64)
        self._hist_counts_yatzy = np.zeros(1, dtype=np.int64)
        self._hist_counts_non_yatzy = np.zeros(1, dtype=np.int64)
        self._hist_counts_bonus = np.zeros(1, dtype=np.int64)
        self._hist_counts_no_bonus = np.zeros(1, dtype=np.int64)
        self._hist_counts_yatzy_bonus = np.zeros(1, dtype=np.int64)
        self._hist_counts_yatzy_no_bonus = np.zeros(1, dtype=np.int64)
        self._hist_counts_no_yatzy_bonus = np.zeros(1, dtype=np.int64)
        self._hist_counts_no_yatzy_no_bonus = np.zeros(1, dtype=np.int64)
        self.min_score = None
        self.max_score = None

    def update(self, results, stats0, stats1, stats2, bonus_hits, yatzy_flags, bonus_flags):
        if results.size == 0:
            return

        yatzy_flags_arr = np.asarray(yatzy_flags, dtype=np.uint8)
        if yatzy_flags_arr.size != results.size:
            raise ValueError("yatzy_flags must match results in length")
        bonus_flags_arr = np.asarray(bonus_flags, dtype=np.uint8)
        if bonus_flags_arr.size != results.size:
            raise ValueError("bonus_flags must match results in length")

        if self.store_results and self._result_chunks is not None:
            self._result_chunks.append(results.copy())
        if self.store_results and self._yatzy_chunks is not None:
            self._yatzy_chunks.append(yatzy_flags_arr.astype(np.bool_, copy=True))
        if self.store_results and self._bonus_chunks is not None:
            self._bonus_chunks.append(bonus_flags_arr.astype(np.bool_, copy=True))

        results_int64 = results.astype(np.int64)
        chunk_sum = float(results_int64.sum())
        chunk_sum_sq = float(np.dot(results_int64, results_int64))
        chunk_count = int(results.size)

        self.total_count += chunk_count
        self.total_sum += chunk_sum
        self.total_sum_sq += chunk_sum_sq
        yatzy_bool = yatzy_flags_arr.astype(np.bool_)
        bonus_bool = bonus_flags_arr.astype(np.bool_)
        bonus_hits_int = int(bonus_bool.sum())
        if bonus_hits_int != int(bonus_hits):
            raise ValueError("bonus_hits does not match sum of bonus flags for chunk")
        self.total_bonus_hits += bonus_hits_int
        yatzy_hits_int = int(yatzy_bool.sum())
        self.total_yatzy_hits += yatzy_hits_int
        yatzy_and_bonus = int(np.count_nonzero(np.logical_and(yatzy_bool, bonus_bool)))
        yatzy_no_bonus = int(np.count_nonzero(np.logical_and(yatzy_bool, np.logical_not(bonus_bool))))
        no_yatzy_bonus = int(np.count_nonzero(np.logical_and(np.logical_not(yatzy_bool), bonus_bool)))
        no_yatzy_no_bonus = chunk_count - yatzy_and_bonus - yatzy_no_bonus - no_yatzy_bonus
        self.total_yatzy_and_bonus += yatzy_and_bonus
        self.total_yatzy_no_bonus += yatzy_no_bonus
        self.total_no_yatzy_bonus += no_yatzy_bonus
        self.total_no_yatzy_no_bonus += no_yatzy_no_bonus

        self.agg_stats0 += np.asarray(stats0, dtype=np.int64)
        self.agg_stats1 += np.asarray(stats1, dtype=np.int64)
        self.agg_stats2 += np.asarray(stats2, dtype=np.int64)

        chunk_min = int(results.min())
        chunk_max = int(results.max())
        self.min_score = chunk_min if self.min_score is None else min(self.min_score, chunk_min)
        self.max_score = chunk_max if self.max_score is None else max(self.max_score, chunk_max)

        self._expand_histogram(chunk_max)
        chunk_counts = np.bincount(results, minlength=chunk_max + 1).astype(np.int64)
        self._hist_counts[:chunk_max + 1] += chunk_counts

        chunk_counts_yatzy = np.bincount(
            results,
            weights=yatzy_bool.astype(np.float64),
            minlength=chunk_max + 1,
        )
        if chunk_counts_yatzy.size:
            chunk_counts_yatzy = np.rint(chunk_counts_yatzy).astype(np.int64)
        self._hist_counts_yatzy[:chunk_max + 1] += chunk_counts_yatzy
        chunk_counts_non_yatzy = chunk_counts - chunk_counts_yatzy
        self._hist_counts_non_yatzy[:chunk_max + 1] += chunk_counts_non_yatzy

        chunk_counts_bonus = np.bincount(
            results,
            weights=bonus_bool.astype(np.float64),
            minlength=chunk_max + 1,
        )
        if chunk_counts_bonus.size:
            chunk_counts_bonus = np.rint(chunk_counts_bonus).astype(np.int64)
        self._hist_counts_bonus[:chunk_max + 1] += chunk_counts_bonus
        chunk_counts_no_bonus = chunk_counts - chunk_counts_bonus
        self._hist_counts_no_bonus[:chunk_max + 1] += chunk_counts_no_bonus

        yb_weights = np.logical_and(yatzy_bool, bonus_bool).astype(np.float64)
        ynb_weights = np.logical_and(yatzy_bool, np.logical_not(bonus_bool)).astype(np.float64)
        nyb_weights = np.logical_and(np.logical_not(yatzy_bool), bonus_bool).astype(np.float64)
        nynb_weights = np.logical_and(np.logical_not(yatzy_bool), np.logical_not(bonus_bool)).astype(np.float64)

        chunk_counts_yatzy_bonus = np.bincount(results, weights=yb_weights, minlength=chunk_max + 1)
        chunk_counts_yatzy_no_bonus = np.bincount(results, weights=ynb_weights, minlength=chunk_max + 1)
        chunk_counts_no_yatzy_bonus = np.bincount(results, weights=nyb_weights, minlength=chunk_max + 1)
        chunk_counts_no_yatzy_no_bonus = np.bincount(results, weights=nynb_weights, minlength=chunk_max + 1)

        if chunk_counts_yatzy_bonus.size:
            chunk_counts_yatzy_bonus = np.rint(chunk_counts_yatzy_bonus).astype(np.int64)
        if chunk_counts_yatzy_no_bonus.size:
            chunk_counts_yatzy_no_bonus = np.rint(chunk_counts_yatzy_no_bonus).astype(np.int64)
        if chunk_counts_no_yatzy_bonus.size:
            chunk_counts_no_yatzy_bonus = np.rint(chunk_counts_no_yatzy_bonus).astype(np.int64)
        if chunk_counts_no_yatzy_no_bonus.size:
            chunk_counts_no_yatzy_no_bonus = np.rint(chunk_counts_no_yatzy_no_bonus).astype(np.int64)

        self._hist_counts_yatzy_bonus[:chunk_max + 1] += chunk_counts_yatzy_bonus
        self._hist_counts_yatzy_no_bonus[:chunk_max + 1] += chunk_counts_yatzy_no_bonus
        self._hist_counts_no_yatzy_bonus[:chunk_max + 1] += chunk_counts_no_yatzy_bonus
        self._hist_counts_no_yatzy_no_bonus[:chunk_max + 1] += chunk_counts_no_yatzy_no_bonus
        del results_int64

    def finalize_results(self):
        if not self.store_results or not self._result_chunks:
            return None
        if len(self._result_chunks) == 1:
            return self._result_chunks[0]
        return np.concatenate(self._result_chunks)

    def finalize_yatzy_flags(self):
        if not self.store_results or not self._yatzy_chunks:
            return None
        if len(self._yatzy_chunks) == 1:
            return self._yatzy_chunks[0]
        return np.concatenate(self._yatzy_chunks)

    def finalize_bonus_flags(self):
        if not self.store_results or not self._bonus_chunks:
            return None
        if len(self._bonus_chunks) == 1:
            return self._bonus_chunks[0]
        return np.concatenate(self._bonus_chunks)

    @property
    def mean(self):
        if self.total_count == 0:
            return 0.0
        return self.total_sum / self.total_count

    @property
    def std(self):
        if self.total_count == 0:
            return 0.0
        mean_val = self.mean
        variance = (self.total_sum_sq / self.total_count) - mean_val * mean_val
        if variance < 0.0:
            variance = 0.0
        return float(np.sqrt(variance))

    def histogram_values(self):
        if self.total_count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        values = np.nonzero(self._hist_counts)[0]
        counts = self._hist_counts[values]
        return values, counts

    def histogram_values_by_yatzy(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty), (empty, empty)
        values_yatzy = np.nonzero(self._hist_counts_yatzy)[0]
        counts_yatzy = self._hist_counts_yatzy[values_yatzy]
        values_non = np.nonzero(self._hist_counts_non_yatzy)[0]
        counts_non = self._hist_counts_non_yatzy[values_non]
        return (values_yatzy, counts_yatzy), (values_non, counts_non)

    def histogram_values_by_bonus(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty), (empty, empty)
        values_bonus = np.nonzero(self._hist_counts_bonus)[0]
        counts_bonus = self._hist_counts_bonus[values_bonus]
        values_no_bonus = np.nonzero(self._hist_counts_no_bonus)[0]
        counts_no_bonus = self._hist_counts_no_bonus[values_no_bonus]
        return (values_bonus, counts_bonus), (values_no_bonus, counts_no_bonus)

    def histogram_values_by_yatzy_and_bonus(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return {
                (True, True): (empty, empty),
                (True, False): (empty, empty),
                (False, True): (empty, empty),
                (False, False): (empty, empty),
            }
        def _extract(arr):
            values = np.nonzero(arr)[0]
            counts = arr[values]
            return values, counts
        return {
            (True, True): _extract(self._hist_counts_yatzy_bonus),
            (True, False): _extract(self._hist_counts_yatzy_no_bonus),
            (False, True): _extract(self._hist_counts_no_yatzy_bonus),
            (False, False): _extract(self._hist_counts_no_yatzy_no_bonus),
        }

    def _expand_histogram(self, max_value):
        if max_value < len(self._hist_counts):
            return
        new_size = max_value + 1
        def _resize(array):
            new_arr = np.zeros(new_size, dtype=np.int64)
            length = len(array)
            new_arr[:length] = array
            return new_arr
        self._hist_counts = _resize(self._hist_counts)
        self._hist_counts_yatzy = _resize(self._hist_counts_yatzy)
        self._hist_counts_non_yatzy = _resize(self._hist_counts_non_yatzy)
        self._hist_counts_bonus = _resize(self._hist_counts_bonus)
        self._hist_counts_no_bonus = _resize(self._hist_counts_no_bonus)
        self._hist_counts_yatzy_bonus = _resize(self._hist_counts_yatzy_bonus)
        self._hist_counts_yatzy_no_bonus = _resize(self._hist_counts_yatzy_no_bonus)
        self._hist_counts_no_yatzy_bonus = _resize(self._hist_counts_no_yatzy_bonus)
        self._hist_counts_no_yatzy_no_bonus = _resize(self._hist_counts_no_yatzy_no_bonus)


def SimulateRounds(
    count,
    processes=None,
    backend="auto",
    chunk_size=None,
    store_results_threshold=1_000_000_000,
    output_dir="results",
    histogram_bins=20,
    save_plots=True,
    show_plots=False,
    save_run_metadata=True,
):
    if count <= 0:
        raise ValueError("count must be a positive integer")

    backend_normalized = backend.lower()
    if backend_normalized not in {"auto", "python", "numba", "cuda"}:
        raise ValueError("backend must be one of 'auto', 'python', 'numba', or 'cuda'")

    if backend_normalized == "auto":
        if CUDA_AVAILABLE:
            backend_normalized = "cuda"
        elif NUMBA_AVAILABLE:
            backend_normalized = "numba"
        else:
            backend_normalized = "python"

    if backend_normalized == "cuda" and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA backend requested but CUDA is not available.")
    if backend_normalized == "numba" and not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    if chunk_size is None:
        chunk_size = min(count, 10_000_000)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = max(1, min(chunk_size, count))

    store_results = count <= store_results_threshold
    accumulator = _SimulationAccumulator(store_results=store_results)

    if backend_normalized == "cuda":
        def backend_runner(batch_size):
            return _simulate_cuda_backend(batch_size)
        run_label_template = "CUDA threads={units}"
    elif backend_normalized == "numba":
        def backend_runner(batch_size):
            return _simulate_numba_backend(batch_size)
        run_label_template = "Numba parallel (threads={units})"
    else:
        def backend_runner(batch_size):
            return _simulate_python_backend(batch_size, processes)
        run_label_template = "{units} process(es)"

    start_time = time.time()
    run_start_unix = int(start_time)
    processed = 0
    units_value = None

    while processed < count:
        batch_size = min(chunk_size, count - processed)
        chunk_start = time.time()
        (
            results,
            bonus_hits_chunk,
            stats0_chunk,
            stats1_chunk,
            stats2_chunk,
            yatzy_flags_chunk,
            bonus_flags_chunk,
            _,
            units,
        ) = backend_runner(batch_size)
        chunk_elapsed = time.time() - chunk_start
        accumulator.update(
            results,
            stats0_chunk,
            stats1_chunk,
            stats2_chunk,
            bonus_hits_chunk,
            yatzy_flags_chunk,
            bonus_flags_chunk,
        )
        processed += batch_size
        if units_value is None:
            units_value = units
        del results, stats0_chunk, stats1_chunk, stats2_chunk, yatzy_flags_chunk, bonus_flags_chunk

        total_elapsed = time.time() - start_time
        percent_complete = processed / count * 100.0
        chunk_time_str = _format_elapsed(chunk_elapsed)
        chunk_rate = batch_size / chunk_elapsed if chunk_elapsed > 0 else None
        chunk_rate_str = (
            f"{chunk_rate:,.0f} games/s" if chunk_rate is not None and math.isfinite(chunk_rate) else "n/a"
        )
        overall_rate = processed / total_elapsed if total_elapsed > 0 else None
        overall_rate_str = (
            f"{overall_rate:,.0f} games/s" if overall_rate is not None and math.isfinite(overall_rate) else "n/a"
        )
        remaining = count - processed
        eta_seconds = None
        if remaining > 0 and overall_rate is not None and math.isfinite(overall_rate) and overall_rate > 0:
            eta_seconds = remaining / overall_rate
        if remaining <= 0:
            eta_message = "complete"
        elif eta_seconds is not None and math.isfinite(eta_seconds):
            eta_display = _format_duration(eta_seconds)
            completion_str = _format_completion_timestamp(eta_seconds)
            if completion_str:
                eta_message = f"{eta_display} (finish ~ {completion_str})"
            else:
                eta_message = eta_display
        else:
            eta_message = "estimating..."

        print(
            (
                f"Chunk complete: {processed:,}/{count:,} ({percent_complete:.2f}%) | "
                f"chunk {batch_size:,} in {chunk_time_str} ({chunk_rate_str}) | "
                f"overall {overall_rate_str} | ETA {eta_message}"
            ),
            flush=True,
        )

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    run_label = run_label_template.format(units=units_value if units_value is not None else "?")

    mean_val = float(accumulator.mean)
    std_val = float(accumulator.std)
    bonus_probability = accumulator.total_bonus_hits / count * 100.0
    yatzy_probability = accumulator.total_yatzy_hits / count * 100.0
    yatzy_bonus_probability = accumulator.total_yatzy_and_bonus / count * 100.0
    yatzy_no_bonus_probability = accumulator.total_yatzy_no_bonus / count * 100.0
    no_yatzy_bonus_probability = accumulator.total_no_yatzy_bonus / count * 100.0
    no_yatzy_no_bonus_probability = accumulator.total_no_yatzy_no_bonus / count * 100.0

    print(f"\n--- RESULTS ({count} games across {run_label}) ---")
    print(f"\nTime: {elapsed_ms:.2f}ms")
    if not store_results or count > chunk_size:
        print(f"Processed in batches of {chunk_size} games (streaming aggregation).")
    print(f"Bonus hit rate: {bonus_probability:.3f}% ({accumulator.total_bonus_hits}/{count})")
    print(f"Yatzy hit rate: {yatzy_probability:.3f}% ({accumulator.total_yatzy_hits}/{count})")
    print(f"Yatzy + Bonus rate: {yatzy_bonus_probability:.3f}% ({accumulator.total_yatzy_and_bonus}/{count})")
    print(f"Yatzy without bonus rate: {yatzy_no_bonus_probability:.3f}% ({accumulator.total_yatzy_no_bonus}/{count})")
    print(f"No Yatzy + Bonus rate: {no_yatzy_bonus_probability:.3f}% ({accumulator.total_no_yatzy_bonus}/{count})")
    print(f"No Yatzy, No Bonus rate: {no_yatzy_no_bonus_probability:.3f}% ({accumulator.total_no_yatzy_no_bonus}/{count})")

    if isinstance(histogram_bins, bool):
        raise ValueError("histogram_bins cannot be a boolean value")
    if histogram_bins is None:
        histogram_bins_effective = "auto"
    else:
        histogram_bins_effective = histogram_bins
        if isinstance(histogram_bins_effective, (int, np.integer)) and histogram_bins_effective <= 0:
            raise ValueError("histogram_bins must be a positive integer when provided as an int")

    if output_dir:
        output_root = Path(output_dir)
    else:
        output_root = Path.cwd()
    output_root.mkdir(parents=True, exist_ok=True)

    run_basename = f"yatzy_{count}_{run_start_unix}"
    run_dir = output_root / run_basename
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "Category": CATEGORY_NAMES,
        "Roll0_count": accumulator.agg_stats0.tolist(),
        "Roll1_count": accumulator.agg_stats1.tolist(),
        "Roll2_count": accumulator.agg_stats2.tolist(),
    })

    df["Roll0_%"] = df["Roll0_count"] / (count * NUM_ROUNDS) * 100
    df["Roll1_%"] = df["Roll1_count"] / (count * NUM_ROUNDS) * 100
    df["Roll2_%"] = df["Roll2_count"] / (count * NUM_ROUNDS) * 100

    summary = pd.DataFrame([
        {"Category": "AverageScore", "Roll0_count": mean_val},
        {"Category": "Std", "Roll0_count": std_val},
        {"Category": "Bonus%", "Roll0_count": bonus_probability},
        {"Category": "Yatzy%", "Roll0_count": yatzy_probability},
        {"Category": "YatzyCount", "Roll0_count": accumulator.total_yatzy_hits},
        {"Category": "BonusCount", "Roll0_count": accumulator.total_bonus_hits},
        {"Category": "Yatzy+BonusCount", "Roll0_count": accumulator.total_yatzy_and_bonus},
        {"Category": "Yatzy+Bonus%", "Roll0_count": yatzy_bonus_probability},
        {"Category": "YatzyNoBonusCount", "Roll0_count": accumulator.total_yatzy_no_bonus},
        {"Category": "YatzyNoBonus%", "Roll0_count": yatzy_no_bonus_probability},
        {"Category": "NoYatzyBonusCount", "Roll0_count": accumulator.total_no_yatzy_bonus},
        {"Category": "NoYatzyBonus%", "Roll0_count": no_yatzy_bonus_probability},
        {"Category": "NoYatzyNoBonusCount", "Roll0_count": accumulator.total_no_yatzy_no_bonus},
        {"Category": "NoYatzyNoBonus%", "Roll0_count": no_yatzy_no_bonus_probability},
        {"Category": "ElapsedMs", "Roll0_count": elapsed_ms},
    ])
    summary_row_count = len(summary)
    df = pd.concat([df, summary], ignore_index=True)

    data_slice = slice(None, -summary_row_count) if summary_row_count else slice(None)

    csv_path = run_dir / f"{run_basename}_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exported results to: {csv_path}")

    hist_values, hist_counts = accumulator.histogram_values()
    (hist_values_yatzy, hist_counts_yatzy), (hist_values_no_yatzy, hist_counts_no_yatzy) = accumulator.histogram_values_by_yatzy()
    (hist_values_bonus, hist_counts_bonus), (hist_values_no_bonus, hist_counts_no_bonus) = accumulator.histogram_values_by_bonus()
    hist_combo_dict = accumulator.histogram_values_by_yatzy_and_bonus()
    hist_values_yatzy_bonus, hist_counts_yatzy_bonus = hist_combo_dict[(True, True)]
    hist_values_yatzy_no_bonus, hist_counts_yatzy_no_bonus = hist_combo_dict[(True, False)]
    hist_values_no_yatzy_bonus, hist_counts_no_yatzy_bonus = hist_combo_dict[(False, True)]
    hist_values_no_yatzy_no_bonus, hist_counts_no_yatzy_no_bonus = hist_combo_dict[(False, False)]

    def _build_full_counts(values, counts, length):
        expanded = np.zeros(length, dtype=np.int64)
        if values.size:
            indices = np.asarray(values, dtype=np.int64)
            expanded[indices] = np.asarray(counts, dtype=np.int64)
        return expanded

    max_score_candidates = []
    if accumulator.max_score is not None:
        max_score_candidates.append(int(accumulator.max_score))
    for value_array in (
        hist_values,
        hist_values_yatzy,
        hist_values_no_yatzy,
        hist_values_bonus,
        hist_values_no_bonus,
        hist_values_yatzy_bonus,
        hist_values_yatzy_no_bonus,
        hist_values_no_yatzy_bonus,
        hist_values_no_yatzy_no_bonus,
    ):
        if value_array.size:
            max_score_candidates.append(int(value_array.max()))
    max_score = max(max_score_candidates) if max_score_candidates else 0
    if max_score < 0:
        max_score = 0
    score_length = max_score + 1

    distribution_columns = [
        (("Both", "Both"), (hist_values, hist_counts)),
        (("Both", "Yes"), (hist_values_yatzy, hist_counts_yatzy)),
        (("Both", "No"), (hist_values_no_yatzy, hist_counts_no_yatzy)),
        (("Yes", "Both"), (hist_values_bonus, hist_counts_bonus)),
        (("No", "Both"), (hist_values_no_bonus, hist_counts_no_bonus)),
        (("Yes", "Yes"), (hist_values_yatzy_bonus, hist_counts_yatzy_bonus)),
        (("Yes", "No"), (hist_values_no_yatzy_bonus, hist_counts_no_yatzy_bonus)),
        (("No", "Yes"), (hist_values_yatzy_no_bonus, hist_counts_yatzy_no_bonus)),
        (("No", "No"), (hist_values_no_yatzy_no_bonus, hist_counts_no_yatzy_no_bonus)),
    ]

    consolidated_counts = {
        labels: _build_full_counts(values, counts, score_length)
        for labels, (values, counts) in distribution_columns
    }

    consolidated_distribution_path = run_dir / f"{run_basename}_distributions.csv"
    with consolidated_distribution_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Bonus?"] + [labels[0] for labels, _ in distribution_columns])
        writer.writerow(["Yatzy?"] + [labels[1] for labels, _ in distribution_columns])
        writer.writerow([f"Score (0-{max_score})"] + ["Count"] * len(distribution_columns))
        for score in range(score_length):
            writer.writerow([score] + [int(consolidated_counts[labels][score]) for labels, _ in distribution_columns])
    print(f"Saved consolidated distributions to: {consolidated_distribution_path}")

    results_array = accumulator.finalize_results()
    yatzy_flags_array = accumulator.finalize_yatzy_flags()
    bonus_flags_array = accumulator.finalize_bonus_flags()

    if (
        results_array is not None
        and yatzy_flags_array is not None
        and bonus_flags_array is not None
        and results_array.size == yatzy_flags_array.size == bonus_flags_array.size
    ):
        yatzy_bool_store = np.asarray(yatzy_flags_array, dtype=np.bool_)
        bonus_bool_store = np.asarray(bonus_flags_array, dtype=np.bool_)
        results_array_yatzy = results_array[yatzy_bool_store]
        results_array_no_yatzy = results_array[~yatzy_bool_store]
        results_array_bonus = results_array[bonus_bool_store]
        results_array_no_bonus = results_array[~bonus_bool_store]
        combo_yatzy_bonus_mask = np.logical_and(yatzy_bool_store, bonus_bool_store)
        combo_yatzy_no_bonus_mask = np.logical_and(yatzy_bool_store, ~bonus_bool_store)
        combo_no_yatzy_bonus_mask = np.logical_and(~yatzy_bool_store, bonus_bool_store)
        combo_no_yatzy_no_bonus_mask = np.logical_and(~yatzy_bool_store, ~bonus_bool_store)
        results_array_yatzy_bonus = results_array[combo_yatzy_bonus_mask]
        results_array_yatzy_no_bonus = results_array[combo_yatzy_no_bonus_mask]
        results_array_no_yatzy_bonus = results_array[combo_no_yatzy_bonus_mask]
        results_array_no_yatzy_no_bonus = results_array[combo_no_yatzy_no_bonus_mask]
    else:
        results_array_yatzy = None
        results_array_no_yatzy = None
        results_array_bonus = None
        results_array_no_bonus = None
        results_array_yatzy_bonus = None
        results_array_yatzy_no_bonus = None
        results_array_no_yatzy_bonus = None
        results_array_no_yatzy_no_bonus = None

    total_yatzy_games = int(hist_counts_yatzy.sum())
    total_no_yatzy_games = int(hist_counts_no_yatzy.sum())
    total_bonus_games = int(hist_counts_bonus.sum())
    total_no_bonus_games = int(hist_counts_no_bonus.sum())
    total_yatzy_bonus_games = int(hist_counts_yatzy_bonus.sum())
    total_yatzy_no_bonus_games = int(hist_counts_yatzy_no_bonus.sum())
    total_no_yatzy_bonus_games = int(hist_counts_no_yatzy_bonus.sum())
    total_no_yatzy_no_bonus_games = int(hist_counts_no_yatzy_no_bonus.sum())

    def _mean_from_hist(values, counts):
        total = counts.sum()
        if total == 0:
            return None
        return float(np.dot(values.astype(np.float64), counts.astype(np.float64)) / float(total))

    mean_yatzy = _mean_from_hist(hist_values_yatzy, hist_counts_yatzy)
    mean_no_yatzy = _mean_from_hist(hist_values_no_yatzy, hist_counts_no_yatzy)
    mean_bonus = _mean_from_hist(hist_values_bonus, hist_counts_bonus)
    mean_no_bonus = _mean_from_hist(hist_values_no_bonus, hist_counts_no_bonus)
    mean_yatzy_bonus = _mean_from_hist(hist_values_yatzy_bonus, hist_counts_yatzy_bonus)
    mean_yatzy_no_bonus = _mean_from_hist(hist_values_yatzy_no_bonus, hist_counts_yatzy_no_bonus)
    mean_no_yatzy_bonus = _mean_from_hist(hist_values_no_yatzy_bonus, hist_counts_no_yatzy_bonus)
    mean_no_yatzy_no_bonus = _mean_from_hist(hist_values_no_yatzy_no_bonus, hist_counts_no_yatzy_no_bonus)

    figures = []
    saved_plot_paths = []

    fig_prob, ax_prob = plt.subplots(figsize=(10, 5))
    categories_no_summary = df["Category"].iloc[data_slice]
    x = np.arange(len(categories_no_summary))
    width = 0.25

    ax_prob.bar(x - width, df["Roll0_%"].iloc[data_slice], width, label="First roll")
    ax_prob.bar(x, df["Roll1_%"].iloc[data_slice], width, label="Reroll 1")
    ax_prob.bar(x + width, df["Roll2_%"].iloc[data_slice], width, label="Reroll 2")

    ax_prob.set_title("Probability to satisfy category (initial roll to second reroll)")
    ax_prob.set_xlabel("Yatzy category")
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(categories_no_summary.to_list(), rotation=45, ha="right")
    ax_prob.legend()
    ax_prob.grid(alpha=0.3, axis="y")
    fig_prob.tight_layout()

    if save_plots:
        category_plot_path = run_dir / f"{run_basename}_category_probabilities.png"
        fig_prob.savefig(category_plot_path, dpi=300)
        saved_plot_paths.append(category_plot_path)
        print(f"Saved category probability plot to: {category_plot_path}")
    figures.append(fig_prob)

    histogram_specs = [
        (
            "All games",
            results_array,
            hist_values,
            hist_counts,
            mean_val,
            count,
            "score_distribution.png",
        ),
        (
            "Yatzy achieved",
            results_array_yatzy,
            hist_values_yatzy,
            hist_counts_yatzy,
            mean_yatzy,
            total_yatzy_games,
            "score_distribution_yatzy.png",
        ),
        (
            "No Yatzy",
            results_array_no_yatzy,
            hist_values_no_yatzy,
            hist_counts_no_yatzy,
            mean_no_yatzy,
            total_no_yatzy_games,
            "score_distribution_no_yatzy.png",
        ),
        (
            "Bonus achieved",
            results_array_bonus,
            hist_values_bonus,
            hist_counts_bonus,
            mean_bonus,
            total_bonus_games,
            "score_distribution_bonus.png",
        ),
        (
            "No bonus",
            results_array_no_bonus,
            hist_values_no_bonus,
            hist_counts_no_bonus,
            mean_no_bonus,
            total_no_bonus_games,
            "score_distribution_no_bonus.png",
        ),
        (
            "Yatzy + bonus",
            results_array_yatzy_bonus,
            hist_values_yatzy_bonus,
            hist_counts_yatzy_bonus,
            mean_yatzy_bonus,
            total_yatzy_bonus_games,
            "score_distribution_yatzy_bonus.png",
        ),
        (
            "Yatzy without bonus",
            results_array_yatzy_no_bonus,
            hist_values_yatzy_no_bonus,
            hist_counts_yatzy_no_bonus,
            mean_yatzy_no_bonus,
            total_yatzy_no_bonus_games,
            "score_distribution_yatzy_no_bonus.png",
        ),
        (
            "No Yatzy + bonus",
            results_array_no_yatzy_bonus,
            hist_values_no_yatzy_bonus,
            hist_counts_no_yatzy_bonus,
            mean_no_yatzy_bonus,
            total_no_yatzy_bonus_games,
            "score_distribution_no_yatzy_bonus.png",
        ),
        (
            "No Yatzy + no bonus",
            results_array_no_yatzy_no_bonus,
            hist_values_no_yatzy_no_bonus,
            hist_counts_no_yatzy_no_bonus,
            mean_no_yatzy_no_bonus,
            total_no_yatzy_no_bonus_games,
            "score_distribution_no_yatzy_no_bonus.png",
        ),
    ]

    for label, data_array, values_subset, counts_subset, mean_subset, subset_total, filename_suffix in histogram_specs:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        if data_array is not None and data_array.size:
            ax_hist.hist(data_array, bins=histogram_bins_effective, edgecolor="black", alpha=0.7)
        elif counts_subset.size:
            ax_hist.hist(values_subset, bins=histogram_bins_effective, weights=counts_subset, edgecolor="black", alpha=0.7)
        else:
            ax_hist.bar([0], [0], width=0.9, edgecolor="black", alpha=0.7)

        if mean_subset is not None:
            ax_hist.axvline(mean_subset, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_subset:.1f}")
            ax_hist.legend()
        ax_hist.set_title(f"Distribution of Yatzy scores ({label})")
        ax_hist.set_xlabel("Points")
        ax_hist.set_ylabel(f"Games (n={subset_total})")
        ax_hist.grid(alpha=0.3)
        fig_hist.tight_layout()

        if save_plots:
            histogram_plot_path = run_dir / f"{run_basename}_{filename_suffix}"
            fig_hist.savefig(histogram_plot_path, dpi=300)
            saved_plot_paths.append(histogram_plot_path)
            print(f"Saved score distribution plot to: {histogram_plot_path}")
        figures.append(fig_hist)

    if show_plots:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    if save_run_metadata:
        system_info = platform.uname()
        metadata_path = run_dir / f"{run_basename}_metadata.json"
        chunk_size_info = int(chunk_size) if chunk_size is not None else None
        finished_at_utc = datetime.fromtimestamp(
            start_time + elapsed_ms / 1000.0, tz=timezone.utc
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        metadata = {
            "run": {
                "count": int(count),
                "backend_requested": backend,
                "backend_used": backend_normalized,
                "run_label": run_label,
                "processes_requested": processes,
                "chunk_size": chunk_size_info,
                "store_results": bool(store_results),
                "store_results_threshold": int(store_results_threshold),
                "results_stored": bool(results_array is not None),
                "bonus_probability_percent": bonus_probability,
                "bonus_hits": int(accumulator.total_bonus_hits),
                "yatzy_probability_percent": yatzy_probability,
                "yatzy_hits": int(accumulator.total_yatzy_hits),
                "yatzy_bonus_probability_percent": yatzy_bonus_probability,
                "yatzy_bonus_hits": int(accumulator.total_yatzy_and_bonus),
                "yatzy_no_bonus_probability_percent": yatzy_no_bonus_probability,
                "yatzy_no_bonus_hits": int(accumulator.total_yatzy_no_bonus),
                "no_yatzy_bonus_probability_percent": no_yatzy_bonus_probability,
                "no_yatzy_bonus_hits": int(accumulator.total_no_yatzy_bonus),
                "no_yatzy_no_bonus_probability_percent": no_yatzy_no_bonus_probability,
                "no_yatzy_no_bonus_hits": int(accumulator.total_no_yatzy_no_bonus),
                "average_score": mean_val,
                "std_dev": std_val,
                "histogram_bins_requested": histogram_bins,
                "histogram_bins_effective": histogram_bins_effective,
                "elapsed_ms": elapsed_ms,
                "output_directory": str(run_dir),
                "run_timestamp_unix": run_start_unix,
            },
            "timing": {
                "started_at_utc": datetime.fromtimestamp(
                    start_time, tz=timezone.utc
                ).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "finished_at_utc": finished_at_utc,
                "elapsed_ms": elapsed_ms,
                "elapsed_seconds": elapsed_ms / 1000.0,
            },
            "system": {
                "platform": system_info.system,
                "platform_release": system_info.release,
                "platform_version": system_info.version,
                "machine": system_info.machine,
                "processor": system_info.processor,
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": sys.executable,
                "cpu_count": os.cpu_count(),
                "numba_available": NUMBA_AVAILABLE,
                "cuda_available": CUDA_AVAILABLE,
            },
            "artifacts": {
                "stats_csv": str(csv_path),
                "distribution_csv": str(consolidated_distribution_path),
                "plots": [str(path) for path in saved_plot_paths],
            },
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        print(f"Saved metadata to: {metadata_path}")

    print(f"Artifacts saved under: {run_dir}")


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


def _parse_histogram_bins_arg(value):
    lowered = value.lower()
    if lowered in {"auto", "sturges", "fd", "doane", "scott", "stone", "rice", "sqrt"}:
        return lowered
    if lowered in {"none", "null"}:
        return None
    return _parse_positive_int(value)


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run Yatzy simulations and export aggregated statistics."
    )
    parser.add_argument(
        "--count",
        type=_parse_positive_int,
        default=1_000_000,
        help="Number of games to simulate (default: 1,000,000).",
    )
    parser.add_argument(
        "--processes",
        type=_parse_optional_positive_int,
        default=16,
        help="Worker processes for Python backend (default: 16). Use 'auto' to let the program decide.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "python", "numba", "cuda"],
        default="auto",
        help="Simulation backend to use (default: auto).",
    )
    parser.add_argument(
        "--chunk-size",
        type=_parse_optional_positive_int,
        default=None,
        help="Batch size per iteration (default: auto). Use 'auto' to pick a default.",
    )
    parser.add_argument(
        "--store-results-threshold",
        type=_parse_positive_int,
        default=1_000_000_000,
        help="Maximum game count to retain individual scores in memory (default: 1,000,000,000).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where run artifacts are written (default: results).",
    )
    parser.add_argument(
        "--histogram-bins",
        type=_parse_histogram_bins_arg,
        default=20,
        help="Bins setting for the score distribution plot (int, 'auto', 'fd', 'sturges', etc.).",
    )
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        action="store_true",
        default=True,
        help="Persist generated plots to disk (default: enabled).",
    )
    parser.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        help="Disable saving plot images.",
    )
    parser.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        default=False,
        help="Display plots interactively (default: disabled).",
    )
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Disable interactive plot display.",
    )
    parser.add_argument(
        "--save-run-metadata",
        dest="save_run_metadata",
        action="store_true",
        default=True,
        help="Export JSON metadata describing the run (default: enabled).",
    )
    parser.add_argument(
        "--no-save-run-metadata",
        dest="save_run_metadata",
        action="store_false",
        help="Skip writing metadata JSON.",
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
            processes=cli_args.processes,
            backend=cli_args.backend,
            chunk_size=cli_args.chunk_size,
            store_results_threshold=cli_args.store_results_threshold,
            output_dir=cli_args.output_dir,
            histogram_bins=cli_args.histogram_bins,
            save_plots=cli_args.save_plots,
            show_plots=cli_args.show_plots,
            save_run_metadata=cli_args.save_run_metadata,
        )
    finally:
        if cli_args.cleanup:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            _cleanup_pycache(script_directory)
