import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
CATEGORY_PRIORITY = (
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
)


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


def _reroll_keep_most_common_array(dice_array, rng, encoded_roll=None):
    roll_index = _encode_roll(dice_array) if encoded_roll is None else encoded_roll
    keep_count = int(ROLL_KEEP_COUNT[roll_index])
    if keep_count == 5:
        return dice_array.copy()

    generator = rng if rng is not None else np.random.default_rng()
    keep_value = int(ROLL_KEEP_VALUE[roll_index])
    rerolled = np.empty(5, dtype=np.int8)
    rerolled[:keep_count] = keep_value
    rerolled[keep_count:] = generator.integers(1, 7, size=5 - keep_count, dtype=np.int8)
    return rerolled


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


## --- MAIN GAMEPLAY ---

def PlayYatzy(rng=None, stats0=None, stats1=None, stats2=None):
    generator = rng if rng is not None else np.random.default_rng()
    allowed_mask = np.ones(NUM_CATEGORIES, dtype=bool)

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

    total_points = 0
    upper_points = 0

    for round_idx in range(1, NUM_ROUNDS + 1):
        dice0 = RollDice(5, generator)
        idx0 = _encode_roll(dice0)
        stats0 += ROLL_SAT_MATRIX[idx0]

        dice1 = _reroll_keep_most_common_array(dice0, generator, idx0)
        idx1 = _encode_roll(dice1)
        stats1 += ROLL_SAT_MATRIX[idx1]

        dice2 = _reroll_keep_most_common_array(dice1, generator, idx1)
        idx2 = _encode_roll(dice2)
        stats2 += ROLL_SAT_MATRIX[idx2]

        score, chosen_idx = _evaluate_best_category_from_index(idx2, allowed_mask)
        total_points += score
        if chosen_idx < 6:
            upper_points += score
        allowed_mask[chosen_idx] = False

        if debug:
            satisfied0 = [CATEGORY_NAMES[i] for i in ROLL_SAT_LISTS[idx0]]
            satisfied1 = [CATEGORY_NAMES[i] for i in ROLL_SAT_LISTS[idx1]]
            satisfied2 = [CATEGORY_NAMES[i] for i in ROLL_SAT_LISTS[idx2]]
            print(f"Satisfied 0 (Round {round_idx}) | {satisfied0}")
            print(f"Satisfied 1 (Round {round_idx}) | {satisfied1}")
            print(f"Satisfied 2 (Round {round_idx}) | {satisfied2}")
            print(f"Round {round_idx} | {CATEGORY_NAMES[chosen_idx]} | {score}")

    bonus = EvaluateBonus(upper_points)
    total_points += bonus

    return total_points, bonus > 0, stats0, stats1, stats2


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

    for i in range(count):
        points, got_bonus, s0, s1, s2 = PlayYatzy(rng, tmp_stats0, tmp_stats1, tmp_stats2)
        results[i] = points
        if got_bonus:
            bonus_hits += 1
        agg_stats0 += s0
        agg_stats1 += s1
        agg_stats2 += s2

    return results, bonus_hits, agg_stats0, agg_stats1, agg_stats2


def SimulateRounds(count, processes=None):
    if count <= 0:
        raise ValueError("count must be a positive integer")

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

    results_arrays = [chunk[0] for chunk in chunk_results]
    results = results_arrays[0] if len(results_arrays) == 1 else np.concatenate(results_arrays)

    bonus_hits = sum(chunk[1] for chunk in chunk_results)

    agg_stats0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_stats2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    for _, _, s0, s1, s2 in chunk_results:
        agg_stats0 += s0
        agg_stats1 += s1
        agg_stats2 += s2

    mean_val = float(np.mean(results))
    std_val = float(np.std(results))
    bonus_probability = bonus_hits / count * 100.0

    print(f"\n--- RESULTS ({count} games across {processes} process(es)) ---")
    print(f"\nTime: {elapsed_ms:.2f}ms")

    df = pd.DataFrame({
        "Category": CATEGORY_NAMES,
        "Roll0_count": agg_stats0.tolist(),
        "Roll1_count": agg_stats1.tolist(),
        "Roll2_count": agg_stats2.tolist(),
    })

    df["Roll0_%"] = df["Roll0_count"] / (count * NUM_ROUNDS) * 100
    df["Roll1_%"] = df["Roll1_count"] / (count * NUM_ROUNDS) * 100
    df["Roll2_%"] = df["Roll2_count"] / (count * NUM_ROUNDS) * 100

    summary = pd.DataFrame([
        {"Category": "AverageScore", "Roll0_count": mean_val},
        {"Category": "Std", "Roll0_count": std_val},
        {"Category": "Bonus%", "Roll0_count": bonus_probability},
    ])
    df = pd.concat([df, summary], ignore_index=True)

    csv_name = f"yatzy_stats_{count}_{int(time.time())}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Exported results to: {csv_name}")

    plt.figure(figsize=(10, 5))
    categories_no_summary = df["Category"][:-3]
    x = np.arange(len(categories_no_summary))
    width = 0.25

    plt.bar(x - width, df["Roll0_%"][:-3], width, label="First roll")
    plt.bar(x, df["Roll1_%"][:-3], width, label="Reroll 1")
    plt.bar(x + width, df["Roll2_%"][:-3], width, label="Reroll 2")

    plt.title("Probability to satisfy category (initial roll to second reroll)")
    plt.xlabel("Yatzy category")
    plt.ylabel("Probability (%)")
    plt.xticks(x, categories_no_summary, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(results, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.1f}")
    plt.title(f"Distribution of Yatzy scores ({count} simulations)")
    plt.xlabel("Points")
    plt.ylabel("Games")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SimulateRounds(100000, processes=os.cpu_count())