import argparse
import csv
import json
import math
import os
import platform
import shutil
import time
from datetime import datetime, timezone, timedelta
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


_HISTOGRAM_KEYS = (
    "all",
    "yatzy",
    "no_yatzy",
    "bonus",
    "no_bonus",
    "yatzy_bonus",
    "yatzy_no_bonus",
    "no_yatzy_bonus",
    "no_yatzy_no_bonus",
)


_HISTOGRAM_SPECS = [
    ("all", "All games", ("Both", "Both"), "score_distribution.png"),
    ("yatzy", "Yatzy achieved", ("Both", "Yes"), "score_distribution_yatzy.png"),
    ("no_yatzy", "No Yatzy", ("Both", "No"), "score_distribution_no_yatzy.png"),
    ("bonus", "Bonus achieved", ("Yes", "Both"), "score_distribution_bonus.png"),
    ("no_bonus", "No bonus", ("No", "Both"), "score_distribution_no_bonus.png"),
    ("yatzy_bonus", "Yatzy + bonus", ("Yes", "Yes"), "score_distribution_yatzy_bonus.png"),
    ("yatzy_no_bonus", "Yatzy without bonus", ("Yes", "No"), "score_distribution_yatzy_no_bonus.png"),
    ("no_yatzy_bonus", "No Yatzy + bonus", ("No", "Yes"), "score_distribution_no_yatzy_bonus.png"),
    ("no_yatzy_no_bonus", "No Yatzy + no bonus", ("No", "No"), "score_distribution_no_yatzy_no_bonus.png"),
]


class _HistogramStore:
    """Utility to maintain aligned histograms for different boolean slices."""

    def __init__(self, keys):
        self._counts = {key: np.zeros(1, dtype=np.int64) for key in keys}

    def ensure_capacity(self, max_score):
        required = max_score + 1
        for key, arr in self._counts.items():
            if arr.size <= max_score:
                expanded = np.zeros(required, dtype=np.int64)
                expanded[: arr.size] = arr
                self._counts[key] = expanded

    def update(self, scores, masks):
        if scores.size == 0:
            return
        max_score = int(scores.max())
        self.ensure_capacity(max_score)
        for key, mask in masks.items():
            counts = self._counts[key]
            weights = None if mask is None else mask.astype(np.int64, copy=False)
            updated = np.bincount(scores, weights=weights, minlength=counts.size)
            if updated.dtype != np.int64:
                updated = np.rint(updated).astype(np.int64)
            counts[: updated.size] += updated

    def values_counts(self, key):
        counts = self._counts[key]
        values = np.nonzero(counts)[0]
        return values, counts[values]

    def copy_counts(self, key):
        return self._counts[key].copy()


class _SimulationAccumulator:
    """Aggregate simulation statistics without storing every game result."""

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
        self._histograms = _HistogramStore(_HISTOGRAM_KEYS)
        self.min_score = None
        self.max_score = None

    def _store_optional_chunks(self, scores, yatzy_flags, bonus_flags):
        if not self.store_results:
            return
        if self._result_chunks is not None:
            self._result_chunks.append(np.asarray(scores, dtype=np.int32).copy())
        if self._yatzy_chunks is not None:
            self._yatzy_chunks.append(np.asarray(yatzy_flags, dtype=bool).copy())
        if self._bonus_chunks is not None:
            self._bonus_chunks.append(np.asarray(bonus_flags, dtype=bool).copy())

    def update(self, results, stats0, stats1, stats2, bonus_hits, yatzy_flags, bonus_flags):
        scores_src = np.asarray(results)
        if scores_src.size == 0:
            return
        if scores_src.ndim != 1:
            raise ValueError("results must be one-dimensional")

        yatzy_flags_arr = np.asarray(yatzy_flags, dtype=np.uint8)
        bonus_flags_arr = np.asarray(bonus_flags, dtype=np.uint8)
        if yatzy_flags_arr.size != scores_src.size or bonus_flags_arr.size != scores_src.size:
            raise ValueError("flag arrays must match results in length")

        self._store_optional_chunks(scores_src, yatzy_flags_arr, bonus_flags_arr)

        scores = scores_src.astype(np.int64, copy=False)
        chunk_sum = float(scores.sum())
        chunk_sum_sq = float(np.dot(scores, scores))
        chunk_count = int(scores.size)
        self.total_count += chunk_count
        self.total_sum += chunk_sum
        self.total_sum_sq += chunk_sum_sq

        yatzy_mask = yatzy_flags_arr.astype(bool, copy=False)
        bonus_mask = bonus_flags_arr.astype(bool, copy=False)
        not_yatzy = ~yatzy_mask
        not_bonus = ~bonus_mask
        combo_yatzy_bonus = yatzy_mask & bonus_mask
        combo_yatzy_no_bonus = yatzy_mask & not_bonus
        combo_no_yatzy_bonus = not_yatzy & bonus_mask
        combo_no_yatzy_no_bonus = not_yatzy & not_bonus

        computed_bonus_hits = int(bonus_mask.sum())
        if int(bonus_hits) != computed_bonus_hits:
            raise ValueError("bonus_hits does not match sum of bonus flags")

        self.total_bonus_hits += computed_bonus_hits
        self.total_yatzy_hits += int(yatzy_mask.sum())
        self.total_yatzy_and_bonus += int(combo_yatzy_bonus.sum())
        self.total_yatzy_no_bonus += int(combo_yatzy_no_bonus.sum())
        self.total_no_yatzy_bonus += int(combo_no_yatzy_bonus.sum())
        self.total_no_yatzy_no_bonus += int(combo_no_yatzy_no_bonus.sum())

        self.agg_stats0 += np.asarray(stats0, dtype=np.int64)
        self.agg_stats1 += np.asarray(stats1, dtype=np.int64)
        self.agg_stats2 += np.asarray(stats2, dtype=np.int64)

        chunk_min = int(scores.min())
        chunk_max = int(scores.max())
        self.min_score = chunk_min if self.min_score is None else min(self.min_score, chunk_min)
        self.max_score = chunk_max if self.max_score is None else max(self.max_score, chunk_max)

        histogram_masks = {
            "all": None,
            "yatzy": yatzy_mask,
            "no_yatzy": not_yatzy,
            "bonus": bonus_mask,
            "no_bonus": not_bonus,
            "yatzy_bonus": combo_yatzy_bonus,
            "yatzy_no_bonus": combo_yatzy_no_bonus,
            "no_yatzy_bonus": combo_no_yatzy_bonus,
            "no_yatzy_no_bonus": combo_no_yatzy_no_bonus,
        }
        self._histograms.update(scores, histogram_masks)

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
            empty = np.array([], dtype=np.int64)
            return empty, empty
        return self._histograms.values_counts("all")

    def histogram_values_by_yatzy(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty), (empty, empty)
        return (
            self._histograms.values_counts("yatzy"),
            self._histograms.values_counts("no_yatzy"),
        )

    def histogram_values_by_bonus(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return (empty, empty), (empty, empty)
        return (
            self._histograms.values_counts("bonus"),
            self._histograms.values_counts("no_bonus"),
        )

    def histogram_values_by_yatzy_and_bonus(self):
        if self.total_count == 0:
            empty = np.array([], dtype=np.int64)
            return {
                (True, True): (empty, empty),
                (True, False): (empty, empty),
                (False, True): (empty, empty),
                (False, False): (empty, empty),
            }
        return {
            (True, True): self._histograms.values_counts("yatzy_bonus"),
            (True, False): self._histograms.values_counts("yatzy_no_bonus"),
            (False, True): self._histograms.values_counts("no_yatzy_bonus"),
            (False, False): self._histograms.values_counts("no_yatzy_no_bonus"),
        }

    def histogram_full_counts(self):
        return {key: self._histograms.copy_counts(key) for key in _HISTOGRAM_KEYS}


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


def _build_stats_dataframe(accumulator, count, mean_val, std_val, elapsed_ms):
    base_df = pd.DataFrame({
        "Category": CATEGORY_NAMES,
        "Roll0_count": accumulator.agg_stats0.tolist(),
        "Roll1_count": accumulator.agg_stats1.tolist(),
        "Roll2_count": accumulator.agg_stats2.tolist(),
    })
    roll_total = count * NUM_ROUNDS
    base_df["Roll0_%"] = base_df["Roll0_count"] / roll_total * 100
    base_df["Roll1_%"] = base_df["Roll1_count"] / roll_total * 100
    base_df["Roll2_%"] = base_df["Roll2_count"] / roll_total * 100

    summary_rows = [
        {"Category": "AverageScore", "Roll0_count": mean_val},
        {"Category": "Std", "Roll0_count": std_val},
        {"Category": "Bonus%", "Roll0_count": accumulator.total_bonus_hits / count * 100.0},
        {"Category": "Yatzy%", "Roll0_count": accumulator.total_yatzy_hits / count * 100.0},
        {"Category": "YatzyCount", "Roll0_count": accumulator.total_yatzy_hits},
        {"Category": "BonusCount", "Roll0_count": accumulator.total_bonus_hits},
        {"Category": "Yatzy+BonusCount", "Roll0_count": accumulator.total_yatzy_and_bonus},
        {"Category": "Yatzy+Bonus%", "Roll0_count": accumulator.total_yatzy_and_bonus / count * 100.0},
        {"Category": "YatzyNoBonusCount", "Roll0_count": accumulator.total_yatzy_no_bonus},
        {"Category": "YatzyNoBonus%", "Roll0_count": accumulator.total_yatzy_no_bonus / count * 100.0},
        {"Category": "NoYatzyBonusCount", "Roll0_count": accumulator.total_no_yatzy_bonus},
        {"Category": "NoYatzyBonus%", "Roll0_count": accumulator.total_no_yatzy_bonus / count * 100.0},
        {"Category": "NoYatzyNoBonusCount", "Roll0_count": accumulator.total_no_yatzy_no_bonus},
        {"Category": "NoYatzyNoBonus%", "Roll0_count": accumulator.total_no_yatzy_no_bonus / count * 100.0},
        {"Category": "ElapsedMs", "Roll0_count": elapsed_ms},
    ]
    summary_df = pd.DataFrame(summary_rows)
    combined = pd.concat([base_df, summary_df], ignore_index=True)
    return combined, len(summary_rows)


def _collect_histogram_slices(accumulator):
    slices = {"all": accumulator.histogram_values()}
    slices["yatzy"], slices["no_yatzy"] = accumulator.histogram_values_by_yatzy()
    slices["bonus"], slices["no_bonus"] = accumulator.histogram_values_by_bonus()
    combo = accumulator.histogram_values_by_yatzy_and_bonus()
    slices["yatzy_bonus"] = combo[(True, True)]
    slices["yatzy_no_bonus"] = combo[(True, False)]
    slices["no_yatzy_bonus"] = combo[(False, True)]
    slices["no_yatzy_no_bonus"] = combo[(False, False)]
    return slices


def _extract_stored_subsets(accumulator):
    results = accumulator.finalize_results()
    yatzy_flags = accumulator.finalize_yatzy_flags()
    bonus_flags = accumulator.finalize_bonus_flags()
    if (
        results is None
        or yatzy_flags is None
        or bonus_flags is None
        or results.size == 0
    ):
        return {}

    scores = np.asarray(results)
    yatzy_mask = np.asarray(yatzy_flags, dtype=bool)
    bonus_mask = np.asarray(bonus_flags, dtype=bool)
    not_yatzy = ~yatzy_mask
    not_bonus = ~bonus_mask

    return {
        "all": scores,
        "yatzy": scores[yatzy_mask],
        "no_yatzy": scores[not_yatzy],
        "bonus": scores[bonus_mask],
        "no_bonus": scores[not_bonus],
        "yatzy_bonus": scores[yatzy_mask & bonus_mask],
        "yatzy_no_bonus": scores[yatzy_mask & not_bonus],
        "no_yatzy_bonus": scores[not_yatzy & bonus_mask],
        "no_yatzy_no_bonus": scores[not_yatzy & not_bonus],
    }


def _max_observed_score(counts_map):
    candidates = []
    for counts in counts_map.values():
        if counts.size == 0:
            continue
        nonzero = np.nonzero(counts)[0]
        if nonzero.size:
            candidates.append(int(nonzero.max()))
    return max(candidates) if candidates else 0


def _write_distribution_csv(run_dir, run_basename, counts_map, max_score):
    path = run_dir / f"{run_basename}_distributions.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Bonus?"] + [labels[0] for _, _, labels, _ in _HISTOGRAM_SPECS])
        writer.writerow(["Yatzy?"] + [labels[1] for _, _, labels, _ in _HISTOGRAM_SPECS])
        writer.writerow([f"Score (0-{max_score})"] + ["Count"] * len(_HISTOGRAM_SPECS))
        for score in range(max_score + 1):
            row = [score]
            for key, _, _, _ in _HISTOGRAM_SPECS:
                counts = counts_map[key]
                value = int(counts[score]) if score < counts.size else 0
                row.append(value)
            writer.writerow(row)
    return path


def _histogram_mean(values, counts):
    if counts.size == 0:
        return None
    total = counts.sum()
    if total == 0:
        return None
    return float(np.dot(values.astype(np.float64), counts.astype(np.float64)) / float(total))


def _plot_category_probabilities(df, data_slice, run_dir, run_basename, save_plots):
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = df["Category"].iloc[data_slice]
    positions = np.arange(len(categories))
    width = 0.25

    ax.bar(positions - width, df["Roll0_%"].iloc[data_slice], width, label="First roll")
    ax.bar(positions, df["Roll1_%"].iloc[data_slice], width, label="Reroll 1")
    ax.bar(positions + width, df["Roll2_%"].iloc[data_slice], width, label="Reroll 2")

    ax.set_title("Probability to satisfy category (initial roll to second reroll)")
    ax.set_xlabel("Yatzy category")
    ax.set_ylabel("Probability (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(categories.to_list(), rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    saved_path = None
    if save_plots:
        saved_path = run_dir / f"{run_basename}_category_probabilities.png"
        fig.savefig(saved_path, dpi=300)
    return fig, saved_path


def _plot_score_histograms(
    run_dir,
    run_basename,
    histogram_bins,
    stored_subsets,
    histogram_slices,
    save_plots,
):
    figures = []
    saved_paths = []
    for key, title, _, filename in _HISTOGRAM_SPECS:
        values, counts = histogram_slices[key]
        data_array = stored_subsets.get(key)
        subset_total = int(data_array.size) if data_array is not None else int(counts.sum())
        mean_val = _histogram_mean(values, counts)

        fig, ax = plt.subplots(figsize=(8, 4))
        if data_array is not None and data_array.size:
            ax.hist(data_array, bins=histogram_bins, edgecolor="black", alpha=0.7)
        elif counts.size:
            ax.hist(values, bins=histogram_bins, weights=counts, edgecolor="black", alpha=0.7)
        else:
            ax.bar([0], [0], width=0.9, edgecolor="black", alpha=0.7)

        if mean_val is not None:
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.1f}")
            ax.legend()

        ax.set_title(f"Distribution of Yatzy scores ({title})")
        ax.set_xlabel("Points")
        ax.set_ylabel(f"Games (n={subset_total})")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_plots:
            path = run_dir / f"{run_basename}_{filename}"
            fig.savefig(path, dpi=300)
            saved_paths.append(path)
        figures.append(fig)

    return figures, saved_paths


def _write_metadata_file(run_dir, run_basename, metadata):
    metadata_path = run_dir / f"{run_basename}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return metadata_path


def SimulateRounds(
    count,
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

    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for simulation but is not available.")

    if chunk_size is None:
        chunk_size = min(count, 10_000_000)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = max(1, min(chunk_size, count))

    store_results = count <= store_results_threshold
    accumulator = _SimulationAccumulator(store_results=store_results)

    simulate_batch = _simulate_numba_backend
    run_label_template = "Numba parallel (threads={units})"

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
        ) = simulate_batch(batch_size)
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

        progress_message = _format_progress_message(processed, count, batch_size, chunk_elapsed, start_time)
        print(progress_message, flush=True)

        del results, stats0_chunk, stats1_chunk, stats2_chunk, yatzy_flags_chunk, bonus_flags_chunk

    elapsed_ms = (time.time() - start_time) * 1000.0
    run_label = run_label_template.format(units=units_value if units_value is not None else "?")

    mean_val = float(accumulator.mean)
    std_val = float(accumulator.std)

    probability_counts = {
        "bonus": accumulator.total_bonus_hits,
        "yatzy": accumulator.total_yatzy_hits,
        "yatzy_bonus": accumulator.total_yatzy_and_bonus,
        "yatzy_no_bonus": accumulator.total_yatzy_no_bonus,
        "no_yatzy_bonus": accumulator.total_no_yatzy_bonus,
        "no_yatzy_no_bonus": accumulator.total_no_yatzy_no_bonus,
    }
    probability_percent = {key: value / count * 100.0 for key, value in probability_counts.items()}

    print(f"\n--- RESULTS ({count} games across {run_label}) ---")
    print(f"\nTime: {elapsed_ms:.2f}ms")
    if not store_results or count > chunk_size:
        print(f"Processed in batches of {chunk_size} games (streaming aggregation).")

    for label, key in [
        ("Bonus hit rate", "bonus"),
        ("Yatzy hit rate", "yatzy"),
        ("Yatzy + Bonus rate", "yatzy_bonus"),
        ("Yatzy without bonus rate", "yatzy_no_bonus"),
        ("No Yatzy + Bonus rate", "no_yatzy_bonus"),
        ("No Yatzy, No Bonus rate", "no_yatzy_no_bonus"),
    ]:
        hits = probability_counts[key]
        percent = probability_percent[key]
        print(f"{label}: {percent:.3f}% ({hits}/{count})")

    if isinstance(histogram_bins, bool):
        raise ValueError("histogram_bins cannot be a boolean value")
    if histogram_bins is None:
        histogram_bins_effective = "auto"
    else:
        histogram_bins_effective = histogram_bins
        if isinstance(histogram_bins_effective, (int, np.integer)) and histogram_bins_effective <= 0:
            raise ValueError("histogram_bins must be a positive integer when provided as an int")

    output_root = Path(output_dir) if output_dir else Path.cwd()
    output_root.mkdir(parents=True, exist_ok=True)

    run_basename = f"yatzy_{count}_{run_start_unix}"
    run_dir = output_root / run_basename
    run_dir.mkdir(parents=True, exist_ok=True)

    stats_df, summary_row_count = _build_stats_dataframe(accumulator, count, mean_val, std_val, elapsed_ms)
    data_slice = slice(None, -summary_row_count) if summary_row_count else slice(None)

    csv_path = run_dir / f"{run_basename}_stats.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"Exported results to: {csv_path}")

    histogram_slices = _collect_histogram_slices(accumulator)
    histogram_counts_map = accumulator.histogram_full_counts()
    max_score = _max_observed_score(histogram_counts_map)

    distribution_path = _write_distribution_csv(run_dir, run_basename, histogram_counts_map, max_score)
    print(f"Saved consolidated distributions to: {distribution_path}")

    stored_subsets = _extract_stored_subsets(accumulator)
    results_array = stored_subsets.get("all")

    figures = []
    saved_plot_paths = []

    fig_prob, prob_path = _plot_category_probabilities(stats_df, data_slice, run_dir, run_basename, save_plots)
    figures.append(fig_prob)
    if prob_path is not None:
        saved_plot_paths.append(prob_path)
        print(f"Saved category probability plot to: {prob_path}")

    hist_figures, hist_paths = _plot_score_histograms(
        run_dir,
        run_basename,
        histogram_bins_effective,
        stored_subsets,
        histogram_slices,
        save_plots,
    )
    figures.extend(hist_figures)
    for path in hist_paths:
        saved_plot_paths.append(path)
        print(f"Saved score distribution plot to: {path}")

    if show_plots:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    if save_run_metadata:
        system_info = platform.uname()
        finished_at_utc = datetime.fromtimestamp(
            start_time + elapsed_ms / 1000.0, tz=timezone.utc
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        metadata = {
            "run": {
                "count": int(count),
                "run_label": run_label,
                "numba_threads_used": int(units_value) if units_value is not None else None,
                "chunk_size": int(chunk_size),
                "store_results": bool(store_results),
                "store_results_threshold": int(store_results_threshold),
                "results_stored": results_array is not None,
                "bonus_probability_percent": probability_percent["bonus"],
                "bonus_hits": int(probability_counts["bonus"]),
                "yatzy_probability_percent": probability_percent["yatzy"],
                "yatzy_hits": int(probability_counts["yatzy"]),
                "yatzy_bonus_probability_percent": probability_percent["yatzy_bonus"],
                "yatzy_bonus_hits": int(probability_counts["yatzy_bonus"]),
                "yatzy_no_bonus_probability_percent": probability_percent["yatzy_no_bonus"],
                "yatzy_no_bonus_hits": int(probability_counts["yatzy_no_bonus"]),
                "no_yatzy_bonus_probability_percent": probability_percent["no_yatzy_bonus"],
                "no_yatzy_bonus_hits": int(probability_counts["no_yatzy_bonus"]),
                "no_yatzy_no_bonus_probability_percent": probability_percent["no_yatzy_no_bonus"],
                "no_yatzy_no_bonus_hits": int(probability_counts["no_yatzy_no_bonus"]),
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
            },
            "artifacts": {
                "stats_csv": str(csv_path),
                "distribution_csv": str(distribution_path),
                "plots": [str(path) for path in saved_plot_paths],
            },
        }
        metadata_path = _write_metadata_file(run_dir, run_basename, metadata)
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
        "--chunk-size",
        type=_parse_optional_positive_int,
        default=None,
        help="Batch size per iteration (default: auto). Use 'auto' to pick a default.",
    )
    parser.add_argument(
        "--store-results-threshold",
        type=_parse_positive_int,
        default=10_000_000,
        help="Maximum game count to retain individual scores in memory (default: 10,000,000).",
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
