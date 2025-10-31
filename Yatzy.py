import random
import argparse
from collections import Counter


def roll_dice(n=5):
    return [random.randint(1, 6) for _ in range(n)]


def choose_keep_value(dice):
    counts = Counter(dice)
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    return max(candidates)


def reroll_to_keep_value(dice, keep_value):
    return [d if d == keep_value else random.randint(1, 6) for d in dice]


def score_scandinavian_yatzy(dice):
    counts = Counter(dice)
    scores = {}

    # Upper section
    for v in range(1, 7):
        s = v * counts.get(v, 0)
        if s > 0:
            scores[f"{v}s"] = s

    # One Pair (highest pair)
    pairs = sorted([v for v, c in counts.items() if c >= 2], reverse=True)
    if pairs:
        scores["One Pair"] = 2 * pairs[0]

    # Two Pair (two distinct pairs)
    if len(pairs) >= 2:
        scores["Two Pair"] = 2 * (pairs[0] + pairs[1])

    # Three of a kind
    threes = sorted([v for v, c in counts.items() if c >= 3], reverse=True)
    if threes:
        scores["Three of a Kind"] = 3 * threes[0]

    # Four of a kind
    fours = sorted([v for v, c in counts.items() if c >= 4], reverse=True)
    if fours:
        scores["Four of a Kind"] = 4 * fours[0]

    # Small Straight (1-2-3-4-5)
    if set(dice) == {1, 2, 3, 4, 5}:
        scores["Small Straight"] = 15

    # Large Straight (2-3-4-5-6)
    if set(dice) == {2, 3, 4, 5, 6}:
        scores["Large Straight"] = 20

    # Full House (3 of a kind + a pair, different values), score is sum of all dice
    has_three = [v for v, c in counts.items() if c >= 3]
    has_pair = [v for v, c in counts.items() if c >= 2]
    full_house = False
    for t in has_three:
        for p in has_pair:
            if p != t and counts[t] >= 3 and counts[p] >= 2:
                full_house = True
                break
        if full_house:
            break
    if full_house:
        scores["Full House"] = sum(dice)

    # Chance
    scores["Chance"] = sum(dice)

    # Yatzy (five of a kind)
    if 5 in counts.values():
        scores["Yatzy"] = 50

    return scores


def choose_best_category(scores):
    # Tie-breaker preference (higher is preferred)
    preference = {
        "Yatzy": 100,
        "Four of a Kind": 90,
        "Full House": 80,
        "Large Straight": 75,
        "Small Straight": 70,
        "Three of a Kind": 60,
        "Two Pair": 50,
        "One Pair": 40,
        "6s": 36,
        "5s": 35,
        "4s": 34,
        "3s": 33,
        "2s": 32,
        "1s": 31,
        "Chance": 10,
    }
    return max(scores.items(), key=lambda kv: (kv[1], preference.get(kv[0], 0)))


def simulate_round(seed=None):
    if seed is not None:
        random.seed(seed)

    initial = roll_dice(5)

    # First reroll: keep the most common value (highest value on tie)
    keep1 = choose_keep_value(initial)
    after_r1 = reroll_to_keep_value(initial, keep1)

    # Second reroll: recalculate keep choice on the new dice
    keep2 = choose_keep_value(after_r1)
    final = reroll_to_keep_value(after_r1, keep2)

    scores = score_scandinavian_yatzy(final)
    best_category, best_score = choose_best_category(scores)

    return {
        "initial": initial,
        "keep_after_r1": keep1,
        "after_r1": after_r1,
        "keep_after_r2": keep2,
        "final": final,
        "scores": scores,
        "best_category": best_category,
        "best_score": best_score,
    }


def simulate_rounds(n_rounds=1000, seed=None):
    if seed is not None:
        random.seed(seed)

    category_counts = Counter()
    scores = []

    for _ in range(n_rounds):
        result = simulate_round()
        category_counts[result["best_category"]] += 1
        scores.append(result["best_score"])

    total = sum(scores)
    avg = total / n_rounds if n_rounds else 0.0

    return {
        "rounds": n_rounds,
        "avg_score": avg,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "category_counts": dict(category_counts),
    }


ALL_CATEGORIES = [
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Four of a Kind",
    "Small Straight",
    "Large Straight",
    "Full House",
    "Chance",
    "Yatzy",
]

# Theoretical maximum scores per category (used for choosing where to put 0)
MAX_POSSIBLE = {
    "1s": 5,
    "2s": 10,
    "3s": 15,
    "4s": 20,
    "5s": 25,
    "6s": 30,
    "One Pair": 12,          # 6+6
    "Two Pair": 22,          # 6+6 + 5+5
    "Three of a Kind": 18,   # 6*3
    "Four of a Kind": 24,    # 6*4
    "Small Straight": 15,    # 1+2+3+4+5
    "Large Straight": 20,    # 2+3+4+5+6
    "Full House": 28,        # 6+6+6 + 5+5
    "Chance": 30,            # 6+6+6+6+6 is 30? actually five sixes = 30
    "Yatzy": 50,
}


def choose_category_for_game(final_dice, available_categories):
    # Compute fulfilled (positive) scores for available categories
    all_positive = score_scandinavian_yatzy(final_dice)
    candidates = {k: v for k, v in all_positive.items() if k in available_categories}

    if candidates:
        # Choose highest score; tie-break using same preference as before
        best_cat, best_score = choose_best_category(candidates)
        return best_cat, best_score

    # No available category yields a positive score; pick a category to assign 0
    # Strategy: sacrifice the category with the lowest theoretical maximum; tie-break by name
    zero_cat = min(available_categories, key=lambda c: (MAX_POSSIBLE.get(c, 0), c))
    return zero_cat, 0


def simulate_game(seed=None):
    if seed is not None:
        random.seed(seed)

    remaining = set(ALL_CATEGORIES)
    assigned = {}
    rounds = []

    while remaining:
        # Play one round (roll + 2 rerolls using the simple strategy)
        initial = roll_dice(5)
        keep1 = choose_keep_value(initial)
        after_r1 = reroll_to_keep_value(initial, keep1)
        keep2 = choose_keep_value(after_r1)
        final = reroll_to_keep_value(after_r1, keep2)

        chosen_cat, score = choose_category_for_game(final, remaining)

        assigned[chosen_cat] = score
        remaining.remove(chosen_cat)

        rounds.append({
            "initial": initial,
            "after_r1": after_r1,
            "final": final,
            "chosen_category": chosen_cat,
            "score": score,
        })

    total_score = sum(assigned.values())

    return {
        "rounds_played": len(rounds),
        "total_score": total_score,
        "per_category": assigned,
        "rounds": rounds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Scandinavian Yatzy rounds")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--game", action="store_true", help="Simulate a full game (use each category once)")
    parser.add_argument("--games", type=int, default=0, help="Simulate N full games and summarize results")
    args = parser.parse_args()

    def simulate_games(n_games=100, seed=None):
        if seed is not None:
            random.seed(seed)

        totals = []
        category_sums = {cat: 0 for cat in ALL_CATEGORIES}
        category_zero_counts = {cat: 0 for cat in ALL_CATEGORIES}

        for _ in range(n_games):
            game = simulate_game(seed=None)
            totals.append(game["total_score"])
            for cat, score in game["per_category"].items():
                category_sums[cat] += score
                if score == 0:
                    category_zero_counts[cat] += 1

        avg_total = sum(totals) / n_games if n_games else 0.0

        per_category_avg = {cat: category_sums[cat] / n_games for cat in ALL_CATEGORIES}
        per_category_zero_rate = {cat: category_zero_counts[cat] / n_games for cat in ALL_CATEGORIES}

        return {
            "games": n_games,
            "avg_total": avg_total,
            "min_total": min(totals) if totals else 0,
            "max_total": max(totals) if totals else 0,
            "per_category_avg": per_category_avg,
            "per_category_zero_rate": per_category_zero_rate,
        }

    if args.games and args.games > 0:
        gsum = simulate_games(n_games=args.games, seed=args.seed)
        print(f"Simulated games:   {gsum['games']}")
        print(f"Average total:     {gsum['avg_total']:.3f}")
        print(f"Min total:         {gsum['min_total']}")
        print(f"Max total:         {gsum['max_total']}")
        print("Average per-category (zeros %):")
        for cat in ALL_CATEGORIES:
            avg = gsum["per_category_avg"][cat]
            zr = gsum["per_category_zero_rate"][cat] * 100
            print(f"  - {cat}: {avg:.3f} ({zr:.1f}% zeros)")
    elif args.game:
        game = simulate_game(seed=args.seed)
        print(f"Game total score:  {game['total_score']}")
        print("Category scores:")
        for cat in ALL_CATEGORIES:
            print(f"  - {cat}: {game['per_category'][cat]}")
    elif args.rounds <= 1:
        result = simulate_round(seed=args.seed)
        print(f"Initial roll:      {result['initial']}")
        print(f"Keep value (r1):   {result['keep_after_r1']}")
        print(f"After reroll 1:    {result['after_r1']}")
        print(f"Keep value (r2):   {result['keep_after_r2']}")
        print(f"Final roll:        {result['final']}")
        fulfilled = {k: v for k, v in result["scores"].items() if v > 0}
        print("Fulfilled categories and scores:")
        for k in sorted(fulfilled, key=lambda x: (-fulfilled[x], x)):
            print(f"  - {k}: {fulfilled[k]}")
        print(f"Chosen category:   {result['best_category']} ({result['best_score']})")
    else:
        summary = simulate_rounds(n_rounds=args.rounds, seed=args.seed)
        print(f"Simulated rounds:  {summary['rounds']}")
        print(f"Average score:     {summary['avg_score']:.3f}")
        print(f"Min score:         {summary['min_score']}")
        print(f"Max score:         {summary['max_score']}")
        print("Category selection counts:")
        for cat, cnt in sorted(summary["category_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            rate = cnt / summary["rounds"]
            print(f"  - {cat}: {cnt} ({rate:.2%})")
