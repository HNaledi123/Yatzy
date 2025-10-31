import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

## --- CONFIGURATION ---
debug = False

## --- HELPERS ---

def RollDice(count):
    return np.random.randint(1, 7, count).tolist()

def EvaluateUpperSection(dice, target_value):
    score = 0
    for n in dice:
        if n == target_value:
            score += target_value
    return score

def EvaluateTwoPairs(dice):
    score = 0
    required = 0
    counts = Counter(dice)
    for val, count in counts.items():
        if count == 2 or count == 3:
            required += 1
            score += val * 2
    if required < 2:
        score = 0
    return score

def EvaluateNOfAKind(dice, required_count):
    score = 0
    counts = Counter(dice)
    for val, count in counts.items():
        if count >= required_count:
            if required_count == 5:
                score = 50
            else:
                score = max(score, val * required_count)
    return score

def EvaluateStraight(dice, size):
    score = 0
    if sorted(dice) == list(range(1 + size, 6 + size)):
        score = 15 + size * 5  # small: 15, large: 20
    return score

def EvaluateFullHouse(dice):
    score = 0
    has_pair = False
    has_three = False
    counts = Counter(dice)
    for val, count in counts.items():
        if count == 2:
            has_pair = True
            score += val * 2
        if count == 3:
            has_three = True
            score += val * 3
    if not (has_pair and has_three):
        score = 0
    return score

def EvaluateChance(dice):
    return sum(dice)

def EvaluateBestCategory(dice, allowed):
    score = -1
    chosen = ""

    for face in range(6, 0, -1):
        evaluation = EvaluateUpperSection(dice, face)
        if evaluation > score and ("Upper" + str(face)) in allowed:
            score = evaluation
            chosen = "Upper" + str(face)

    for p in range(5, 1, -1):
        evaluation = EvaluateNOfAKind(dice, p)
        if evaluation > score and ("OfAKind" + str(p)) in allowed:
            score = evaluation
            chosen = "OfAKind" + str(p)

    evaluation = EvaluateTwoPairs(dice)
    if evaluation > score and "TwoPairs" in allowed:
        score = evaluation
        chosen = "TwoPairs"

    for s in (1, 0):  # check Large then Small
        evaluation = EvaluateStraight(dice, s)
        name = "LargeStraight" if s == 1 else "SmallStraight"
        if evaluation > score and name in allowed:
            score = evaluation
            chosen = name

    evaluation = EvaluateFullHouse(dice)
    if evaluation > score and "FullHouse" in allowed:
        score = evaluation
        chosen = "FullHouse"

    evaluation = EvaluateChance(dice)
    if evaluation > score and "Chance" in allowed:
        score = evaluation
        chosen = "Chance"

    return score, chosen

def SatisfiedCategories(dice):
    satisfied = []

    for face in range(6, 0, -1):
        evaluation = EvaluateUpperSection(dice, face)
        if evaluation > 0:
            satisfied.append("Upper" + str(face))

    for p in range(5, 1, -1):
        evaluation = EvaluateNOfAKind(dice, p)
        if evaluation > 0:
            satisfied.append("OfAKind" + str(p))

    evaluation = EvaluateTwoPairs(dice)
    if evaluation > 0:
        satisfied.append("TwoPairs")

    for s in (1, 0):
        evaluation = EvaluateStraight(dice, s)
        if evaluation > 0:
            satisfied.append("LargeStraight" if s == 1 else "SmallStraight")

    evaluation = EvaluateFullHouse(dice)
    if evaluation > 0:
        satisfied.append("FullHouse")

    evaluation = EvaluateChance(dice)
    if evaluation > 0:
        satisfied.append("Chance")

    return satisfied

def EvaluateBonus(points):
    score = 0
    if points >= 63:
        score = 50
    if debug:
        print("Bonus | " + str(score))
    return score

# REROLL STRATEGY:
# On each roll keep the most frequent value(s), reroll the rest.

def RerollKeepMostCommon(dice):
    new_dice = []
    maxcount = 0
    maxval = 0
    counts = Counter(dice)
    for val, count in counts.items():
        if count > maxcount or (count == maxcount and val > maxval):
            maxval = val
            maxcount = count
    i = maxcount
    while i > 0:
        new_dice.append(int(maxval))
        i -= 1
    reroll = RollDice(5 - maxcount)
    for n in reroll:
        new_dice.append(n)
    return new_dice

## --- MAIN ---

def PlayYatzy():
    allowed_categories = [
        "Upper1","Upper2","Upper3","Upper4","Upper5","Upper6",
        "OfAKind2","OfAKind3","OfAKind4","OfAKind5",
        "TwoPairs","SmallStraight","LargeStraight","FullHouse","Chance"
    ]
    points = 0
    upper_points = 0
    round_idx = 1
    stats = {r: 0 for r in allowed_categories}
    stats_reroll1 = stats.copy()
    stats_reroll2 = stats.copy()
    while round_idx < 16:
        dice = RollDice(5)
        dice_r1 = RerollKeepMostCommon(dice)
        dice_r2 = RerollKeepMostCommon(dice_r1)
        evaluation = EvaluateBestCategory(dice_r2, allowed_categories)
        req_eval0 = SatisfiedCategories(dice)
        req_eval1 = SatisfiedCategories(dice_r1)
        req_eval2 = SatisfiedCategories(dice_r2)
        if debug:
            print("Satisfied 0 (Round " + str(round_idx) + ") | " + str(req_eval0))
            print("Satisfied 1 (Round " + str(round_idx) + ") | " + str(req_eval1))
            print("Satisfied 2 (Round " + str(round_idx) + ") | " + str(req_eval2))
            print("Round " + str(round_idx) + " | " + evaluation[1] + " | " + str(evaluation[0]))
        for k, s in stats.items():
            if k in req_eval0:
                s += 1
                stats.update({k: s})
        for k, s in stats_reroll1.items():
            if k in req_eval1:
                s += 1
                stats_reroll1.update({k: s})
        for k, s in stats_reroll2.items():
            if k in req_eval2:
                s += 1
                stats_reroll2.update({k: s})

        points += evaluation[0]
        allowed_categories.remove(evaluation[1])
        if "Upper" in evaluation[1]:
            upper_points += evaluation[0]
        round_idx += 1
    bonus = EvaluateBonus(upper_points)
    points += bonus
    return points, bonus > 0, stats, stats_reroll1, stats_reroll2

def SimulateRounds(count):
    categories = [
        "Upper1","Upper2","Upper3","Upper4","Upper5","Upper6",
        "OfAKind2","OfAKind3","OfAKind4","OfAKind5",
        "TwoPairs","SmallStraight","LargeStraight","FullHouse","Chance"
    ]
    start = time.time()
    results = []
    bonus_hits = 0
    agg_stats0 = {r: 0 for r in categories}
    agg_stats1 = agg_stats0.copy()
    agg_stats2 = agg_stats0.copy()
    for _ in range(count):
        pts, got_bonus, stats0, s1, s2 = PlayYatzy()
        results.append(pts)
        if got_bonus:
            bonus_hits += 1
        for r, v in stats0.items():
            agg_stats0.update({r: agg_stats0[r] + v})
        for r, v in s1.items():
            agg_stats1.update({r: agg_stats1[r] + v})
        for r, v in s2.items():
            agg_stats2.update({r: agg_stats2[r] + v})

    end = time.time()
    time_ms = (end - start) * 1000
    
    # Statistics
    mean_val = np.mean(results)
    std_val = np.std(results)
    bonus_probability = bonus_hits / count * 100

    # Output
    print(f"\n--- RESULTS ({count} games) ---")
    print(f"\nTime: {time_ms}ms")

    # --- CSV EXPORT + PLOTTING ---

    df = pd.DataFrame({
        "Category": categories,
        "Roll0_count": [agg_stats0[r] for r in categories],
        "Roll1_count": [agg_stats1[r] for r in categories],
        "Roll2_count": [agg_stats2[r] for r in categories],
    })

    df["Roll0_%"] = df["Roll0_count"] / count / 15 * 100
    df["Roll1_%"] = df["Roll1_count"] / count / 15 * 100
    df["Roll2_%"] = df["Roll2_count"] / count / 15 * 100

    # Append summary rows
    summary = pd.DataFrame([
        {"Category": "AverageScore", "Roll0_count": mean_val},
        {"Category": "Std", "Roll0_count": std_val},
        {"Category": "Bonus%", "Roll0_count": bonus_probability}
    ])
    df = pd.concat([df, summary], ignore_index=True)

    csv_name = f"yatzy_stats_{count}_{time.time()}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Exported results to: {csv_name}")

    # Plot evolution of probabilities per category (BAR CHART)
    plt.figure(figsize=(10, 5))

    categories_no_summary = df["Category"][:-3]
    x = np.arange(len(categories_no_summary))
    width = 0.25  # bar width

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

SimulateRounds(10000)