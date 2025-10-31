import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
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


def _evaluate_best_category(dice_array, counts, sorted_dice, dice_sum, allowed_mask):
    del counts, sorted_dice, dice_sum
    roll_index = _encode_roll(dice_array)
    return _evaluate_best_category_from_index(roll_index, allowed_mask)


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
            allowed_mask[chosen_idx] = False

        bonus = 50 if upper_points >= 63 else 0
        total_points += bonus
        return total_points, bonus > 0

    @njit(parallel=True, cache=True)
    def _simulate_chunk_numba(count, seed):
        np.random.seed(seed)
        results = np.empty(count, dtype=np.int32)
        bonus_hits = np.empty(count, dtype=np.int8)
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

            total_points, got_bonus = _play_yatzy_numba(
                stats0_local, stats1_local, stats2_local, allowed_mask_local, dice0, dice1, dice2
            )
            results[i] = total_points
            bonus_hits[i] = np.int8(1 if got_bonus else 0)

            for c in range(NUM_CATEGORIES):
                stats0_all[i, c] = stats0_local[c]
                stats1_all[i, c] = stats1_local[c]
                stats2_all[i, c] = stats2_local[c]

        return results, bonus_hits, stats0_all, stats1_all, stats2_all


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
    def _simulate_kernel_cuda(rng_states, results, bonus_hits, stats0_all, stats1_all, stats2_all,
                              roll_score_table, roll_sat_matrix, roll_keep_value, roll_keep_count, category_priority):
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
            allowed_mask_local[best_idx] = False

        bonus = 50 if upper_points >= 63 else 0
        total_points += bonus

        results[idx] = total_points
        bonus_hits[idx] = 1 if bonus > 0 else 0
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

    return total_points, bonus > 0


def _simulate_chunk(task):
    count, seed = task
    rng = np.random.default_rng(seed)

    results = np.empty(count, dtype=np.int32)
    bonus_hits = 0
    agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    tmp_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    allowed_mask = np.empty(NUM_CATEGORIES, dtype=bool)

    for i in range(count):
        points, got_bonus = PlayYatzy(rng, tmp_stats0, tmp_stats1, tmp_stats2, allowed_mask)
        results[i] = points
        if got_bonus:
            bonus_hits += 1
        agg_stats0 += tmp_stats0
        agg_stats1 += tmp_stats1
        agg_stats2 += tmp_stats2

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2


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
    bonus_hits = 0
    offset = 0
    for (scores, chunk_bonus, s0, s1, s2), size in zip(chunk_results, chunk_sizes):
        end = offset + size
        results[offset:end] = scores
        offset = end
        bonus_hits += chunk_bonus
        agg_stats0 += s0
        agg_stats1 += s1
        agg_stats2 += s2

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2, elapsed_ms, processes


def _simulate_numba_backend(count):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    seed_sequence = np.random.SeedSequence()
    seed = int(seed_sequence.generate_state(1)[0])

    start = time.time()
    results, bonus_hits_array, stats0_all, stats1_all, stats2_all = _simulate_chunk_numba(count, seed)
    elapsed_ms = (time.time() - start) * 1000

    agg_stats0 = stats0_all.sum(axis=0, dtype=np.int64)
    agg_stats1 = stats1_all.sum(axis=0, dtype=np.int64)
    agg_stats2 = stats2_all.sum(axis=0, dtype=np.int64)
    bonus_hits = int(bonus_hits_array.sum())

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2, elapsed_ms, get_num_threads()


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
        stats0_dev,
        stats1_dev,
        stats2_dev,
        roll_score_table_dev,
        roll_sat_matrix_dev,
        roll_keep_value_dev,
        roll_keep_count_dev,
        category_priority_dev,
    )
    cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000

    results = results_dev.copy_to_host()
    bonus_hits_array = bonus_hits_dev.copy_to_host()
    stats0_all = stats0_dev.copy_to_host()
    stats1_all = stats1_dev.copy_to_host()
    stats2_all = stats2_dev.copy_to_host()

    agg_stats0 = stats0_all.sum(axis=0, dtype=np.int64)
    agg_stats1 = stats1_all.sum(axis=0, dtype=np.int64)
    agg_stats2 = stats2_all.sum(axis=0, dtype=np.int64)
    bonus_hits = int(bonus_hits_array.sum())

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2, elapsed_ms, total_threads


class _SimulationAccumulator:
    """Aggregate simulation statistics incrementally to avoid storing per-game data."""

    def __init__(self, store_results):
        self.store_results = store_results
        self._result_chunks = [] if store_results else None
        self.total_count = 0
        self.total_sum = 0.0
        self.total_sum_sq = 0.0
        self.total_bonus_hits = 0
        self.agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self.agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self.agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        self._hist_counts = np.zeros(1, dtype=np.int64)
        self.min_score = None
        self.max_score = None

    def update(self, results, stats0, stats1, stats2, bonus_hits):
        if results.size == 0:
            return

        if self.store_results and self._result_chunks is not None:
            self._result_chunks.append(results.copy())

        results_int64 = results.astype(np.int64)
        chunk_sum = float(results_int64.sum())
        chunk_sum_sq = float(np.dot(results_int64, results_int64))
        chunk_count = int(results.size)

        self.total_count += chunk_count
        self.total_sum += chunk_sum
        self.total_sum_sq += chunk_sum_sq
        self.total_bonus_hits += int(bonus_hits)

        self.agg_stats0 += np.asarray(stats0, dtype=np.int64)
        self.agg_stats1 += np.asarray(stats1, dtype=np.int64)
        self.agg_stats2 += np.asarray(stats2, dtype=np.int64)

        chunk_min = int(results.min())
        chunk_max = int(results.max())
        self.min_score = chunk_min if self.min_score is None else min(self.min_score, chunk_min)
        self.max_score = chunk_max if self.max_score is None else max(self.max_score, chunk_max)

        self._expand_histogram(chunk_max)
        chunk_counts = np.bincount(results, minlength=chunk_max + 1)
        self._hist_counts[:chunk_max + 1] += chunk_counts.astype(np.int64)
        del results_int64

    def finalize_results(self):
        if not self.store_results or not self._result_chunks:
            return None
        if len(self._result_chunks) == 1:
            return self._result_chunks[0]
        return np.concatenate(self._result_chunks)

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

    def _expand_histogram(self, max_value):
        if max_value < len(self._hist_counts):
            return
        new_size = max_value + 1
        new_counts = np.zeros(new_size, dtype=np.int64)
        new_counts[:len(self._hist_counts)] = self._hist_counts
        self._hist_counts = new_counts


def SimulateRounds(
    count,
    processes=None,
    backend="auto",
    chunk_size=None,
    store_results_threshold=5_000_000,
    output_dir="results",
    histogram_bins=20,
    save_plots=True,
    show_plots=True,
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
        results, bonus_hits_chunk, stats0_chunk, stats1_chunk, stats2_chunk, _, units = backend_runner(batch_size)
        accumulator.update(results, stats0_chunk, stats1_chunk, stats2_chunk, bonus_hits_chunk)
        processed += batch_size
        if units_value is None:
            units_value = units
        del results, stats0_chunk, stats1_chunk, stats2_chunk

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    run_label = run_label_template.format(units=units_value if units_value is not None else "?")

    mean_val = float(accumulator.mean)
    std_val = float(accumulator.std)
    bonus_probability = accumulator.total_bonus_hits / count * 100.0

    print(f"\n--- RESULTS ({count} games across {run_label}) ---")
    print(f"\nTime: {elapsed_ms:.2f}ms")
    if not store_results or count > chunk_size:
        print(f"Processed in batches of {chunk_size} games (streaming aggregation).")

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
        {"Category": "ElapsedMs", "Roll0_count": elapsed_ms},
    ])
    summary_row_count = len(summary)
    df = pd.concat([df, summary], ignore_index=True)

    data_slice = slice(None, -summary_row_count) if summary_row_count else slice(None)

    csv_path = run_dir / f"{run_basename}_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exported results to: {csv_path}")

    hist_values, hist_counts = accumulator.histogram_values()
    distribution_df = pd.DataFrame({
        "Score": hist_values.tolist(),
        "Count": hist_counts.tolist(),
    })
    distribution_path = run_dir / f"{run_basename}_distribution.csv"
    distribution_df.to_csv(distribution_path, index=False)
    print(f"Saved per-score counts to: {distribution_path}")

    results_array = accumulator.finalize_results()

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

    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    if results_array is not None and results_array.size:
        ax_hist.hist(results_array, bins=histogram_bins_effective, edgecolor="black", alpha=0.7)
    elif hist_counts.size:
        ax_hist.hist(hist_values, bins=histogram_bins_effective, weights=hist_counts, edgecolor="black", alpha=0.7)
    else:
        ax_hist.bar([0], [0], width=0.9, edgecolor="black", alpha=0.7)

    ax_hist.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.1f}")
    ax_hist.set_title(f"Distribution of Yatzy scores ({count} simulations)")
    ax_hist.set_xlabel("Points")
    ax_hist.set_ylabel("Games")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)
    fig_hist.tight_layout()

    if save_plots:
        histogram_plot_path = run_dir / f"{run_basename}_score_distribution.png"
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
                "distribution_csv": str(distribution_path),
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
        default=True,
        help="Display plots interactively (default: enabled).",
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
