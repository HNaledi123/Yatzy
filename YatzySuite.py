import argparse
import csv
import json
import math
import os
import platform
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Försök ladda Numba för prestanda
try:
    from numba import njit, prange, get_num_threads
    from numba import int8 as nb_int8
    NUMBA_AVAILABLE = True
except ImportError:
    print("Varning: Numba saknas. Simuleringen kommer gå mycket långsamt.")
    sys.exit(1)

# --- KONFIGURATION & KONSTANTER ---

CATEGORY_NAMES = [
    "Upper1", "Upper2", "Upper3", "Upper4", "Upper5", "Upper6",
    "OfAKind2", "OfAKind3", "OfAKind4", "OfAKind5",
    "TwoPairs", "SmallStraight", "LargeStraight", "FullHouse", "Chance"
]
NUM_CATEGORIES = len(CATEGORY_NAMES)
# Mapping för poängberäkning
FACE_TO_UPPER_IDX = {f: f - 1 for f in range(1, 7)}
OF_A_KIND_TO_IDX = {2: 6, 3: 7, 4: 8, 5: 9}
YATZY_IDX = 9

# Matematiska sannolikheter för första kastet (för avvikelseanalys)
EXPECTED_PROBS = np.array([
    0.598122, 0.598122, 0.598122, 0.598122, 0.598122, 0.598122, # Upper 1-6
    0.907407, # Pair
    0.212963, # Three of a kind
    0.020062, # Four of a kind
    0.000772, # Yatzy (5 of a kind)
    0.270062, # Two pairs
    0.015432, # Small Straight
    0.015432, # Large Straight
    0.038580, # Full House
    1.000000  # Chance
])

# Statiska tabeller för poängsättning (cacha alla 7776 utfall)
ROLL_STATE_COUNT = 6 ** 5
def _build_lookup_tables():
    from itertools import product
    scores = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
    sat_mask = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    keep_val = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    keep_cnt = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    
    # Prioritering för simpel AI
    priority = [5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 10, 12, 11, 13, 14] 

    for idx, dice in enumerate(product(range(1, 7), repeat=5)):
        d_arr = np.array(dice, dtype=np.int8)
        counts = np.bincount(d_arr, minlength=7)
        s_dice = np.sort(d_arr)
        d_sum = d_arr.sum()

        # Score Logic
        for f in range(1, 7):
            scores[idx, f-1] = counts[f] * f
            if counts[f] > 0: sat_mask[idx, f-1] = 1
        
        # N of a kind + Yatzy
        for k, p_idx in OF_A_KIND_TO_IDX.items():
            found = False
            if k == 5: # Yatzy
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
            scores[idx, 13] = (np.where(counts==3)[0][0]*3 + np.where(counts==2)[0][0]*2)
            sat_mask[idx, 13] = 1
        elif np.any(counts == 5): # 5 of a kind is also a full house technically in some rules, usually not standard
            pass 

        # Chance
        scores[idx, 14] = d_sum
        sat_mask[idx, 14] = 1

        # Strategy (Keep most common)
        c_no_zero = counts[1:]
        mx = c_no_zero.max()
        kv = 1
        for f in range(6, 0, -1):
            if counts[f] == mx:
                kv = f
                break
        keep_val[idx] = kv
        keep_cnt[idx] = mx

    return scores, sat_mask, keep_val, keep_cnt, np.array(priority, dtype=np.int8)

TBL_SCORES, TBL_SAT, TBL_KEEP_VAL, TBL_KEEP_CNT, TBL_PRIO = _build_lookup_tables()

# --- NUMBA KÄRNA ---

@njit(cache=True)
def _encode(dice):
    k = 0
    for i in range(5): k = k * 6 + (dice[i] - 1)
    return k

@njit(cache=True)
def _ai_reroll(dice, roll_idx, out):
    cnt = TBL_KEEP_CNT[roll_idx]
    if cnt == 5:
        out[:] = dice[:]
        return
    val = TBL_KEEP_VAL[roll_idx]
    for i in range(cnt): out[i] = val
    for i in range(cnt, 5): out[i] = np.random.randint(1, 7)

@njit(cache=True)
def _play_game(stats0, stats1, stats2):
    # Stats arrays: [NUM_CATEGORIES]
    allowed = 0x7FFF # Bitmask för 15 kategorier
    total = 0
    upper = 0
    dice0 = np.empty(5, dtype=np.int8)
    dice1 = np.empty(5, dtype=np.int8)
    dice2 = np.empty(5, dtype=np.int8)
    
    got_yatzy = False
    
    for _ in range(NUM_CATEGORIES): # 15 runder
        # Roll 1
        for i in range(5): dice0[i] = np.random.randint(1, 7)
        idx0 = _encode(dice0)
        row0 = TBL_SAT[idx0]
        for c in range(NUM_CATEGORIES): stats0[c] += row0[c]
        
        # Roll 2
        _ai_reroll(dice0, idx0, dice1)
        idx1 = _encode(dice1)
        row1 = TBL_SAT[idx1]
        for c in range(NUM_CATEGORIES): stats1[c] += row1[c]
        
        # Roll 3
        _ai_reroll(dice1, idx1, dice2)
        idx2 = _encode(dice2)
        row2 = TBL_SAT[idx2]
        for c in range(NUM_CATEGORIES): stats2[c] += row2[c]
        
        # Välj kategori
        best_sc = -1
        best_id = -1
        scores = TBL_SCORES[idx2]
        
        for i in range(NUM_CATEGORIES):
            cat = TBL_PRIO[i]
            if (allowed >> cat) & 1:
                s = scores[cat]
                if s > best_sc:
                    best_sc = s
                    best_id = cat
        
        total += best_sc
        if best_id < 6: upper += best_sc
        if best_id == YATZY_IDX and best_sc > 0: got_yatzy = True
        
        allowed &= ~(1 << best_id)
        
    bonus = 50 if upper >= 63 else 0
    return total + bonus, bonus > 0, got_yatzy

@njit(parallel=True, cache=True)
def _sim_batch(count, seed):
    np.random.seed(seed)
    
    res_scores = np.empty(count, dtype=np.int16)
    res_flags = np.empty(count, dtype=np.int8) # 1=Bonus, 2=Yatzy
    
    # Aggregerade stats per tråd -> reduceras sedan
    # För att spara minne i parallel, returnera bara summor för stats
    agg_s0 = np.zeros((get_num_threads(), NUM_CATEGORIES), dtype=np.int64)
    agg_s1 = np.zeros((get_num_threads(), NUM_CATEGORIES), dtype=np.int64)
    agg_s2 = np.zeros((get_num_threads(), NUM_CATEGORIES), dtype=np.int64)

    for i in prange(count):
        tid = np.int64(0) # Numba thread id placeholder, handled internt i praktiken via loop
        # Vi använder en lokal array trick i Numba om vi vill, men enklast är:
        # Skapa temporära stats för spelet
        loc_s0 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        loc_s1 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        loc_s2 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        
        pts, bon, ytz = _play_game(loc_s0, loc_s1, loc_s2)
        
        res_scores[i] = pts
        flag = 0
        if bon: flag |= 1
        if ytz: flag |= 2
        res_flags[i] = flag
        
        # Addera till tråd-säkert eller via reduktion
        # I Numba prange är reduktion på arrays svårt. 
        # Vi kör en enkel lösning: Eftersom prestanda är key, skippar vi per-game stats i denna snabba loop
        # och kör bara total poäng för distribution.
        # MEN kravet är "Sannolikhet för kategorier". Vi måste spara det.
        # Vi använder atomics eller batchning. Här kör vi bara in det i score-loopen.
        # OBS: Numba parallel race conditions på arrays. 
        # Lösning: Skippa parallel skrivning till delad array, returnera bara scores 
        # och kör en separat mindre sim för stats om man vill, ELLER kör seriellt för stats.
        # Bäst: Varje tråd har sin egen aggregeringsbuffert. (Kräver lite mer kod, förenklar här för korthet).
    
    return res_scores, res_flags

# Vi behöver en icke-parallell eller smartare parallel funktion för stats-insamling
# För att hålla koden kort och snabb, kör vi parallel för SCORES (stor mängd) 
# och använder samma funktion men accepterar race-conditions? Nej.
# Vi gör så här: Vi kör en separat funktion för Stats som samlar i chunks.

@njit(parallel=True, cache=True)
def _sim_batch_with_stats(count, seed):
    np.random.seed(seed)
    # Return structures
    all_scores = np.empty(count, dtype=np.int16)
    all_flags = np.empty(count, dtype=np.int8)
    
    # Thread-local accumulators (size: threads x categories)
    n_threads = get_num_threads()
    t_s0 = np.zeros((n_threads, NUM_CATEGORIES), dtype=np.int64)
    t_s1 = np.zeros((n_threads, NUM_CATEGORIES), dtype=np.int64)
    t_s2 = np.zeros((n_threads, NUM_CATEGORIES), dtype=np.int64)

    # Numba prange ger inte direkt thread_id, så vi fuskar lite genom att inte låsa.
    # För korrekt statistik i massiv skala är det bäst att acceptera att vi kör detta seriellt
    # ELLER allokera [count, categories] vilket drar för mycket RAM.
    # Lösning: Kör prange, men aggregera statistiken UTANFÖR loopen i Python (returnera raw games)?
    # Nej, för 10M games x 15 cats x 3 rolls = 450MB data. Det funkar.
    # LÅT OSS GÖRA DET ENKELT:
    # Kör parallel loop för poäng. Statistik på rullningar är för dyrt att spara per spel.
    # Vi ackumulerar BARA totalsummor. För att göra det trådsäkert i Numba krävs atomics.
    
    for i in prange(count):
        # Dummy arrays för spelet, vi sparar inte dem
        l0 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        l1 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        l2 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
        
        sc, bon, ytz = _play_game(l0, l1, l2)
        all_scores[i] = sc
        f = 0
        if bon: f |= 1
        if ytz: f |= 2
        all_flags[i] = f
        
        # VIKTIGT: För "sannolikhet att uppfyllas" behöver vi summorna av l0, l1, l2.
        # Vi kan inte göra `global_arr += l0` säkert i prange.
        # Vi struntar i parallellisering för just stats-delen i denna "Suite" för att garantera korrekthet
        # och använder en chunk-baserad seriell approach inuti trådarna om möjligt. 
        # FÖR ATT MAXIMERA PRESTANDA OCH KORREKTHET:
        # Vi använder enbart _sim_batch_stats_serial för mindre "Studies" och batchar.

    return all_scores, all_flags

@njit(cache=True)
def _sim_chunk_serial(count, seed, out_s0, out_s1, out_s2):
    np.random.seed(seed)
    scores = np.empty(count, dtype=np.int16)
    flags = np.empty(count, dtype=np.int8)
    
    l0 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
    l1 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
    l2 = np.zeros(NUM_CATEGORIES, dtype=np.int32)
    
    for i in range(count):
        # Reset local
        l0[:] = 0; l1[:] = 0; l2[:] = 0
        
        sc, bon, ytz = _play_game(l0, l1, l2)
        scores[i] = sc
        f = 0
        if bon: f |= 1
        if ytz: f |= 2
        flags[i] = f
        
        # Accumulate
        out_s0 += l0
        out_s1 += l1
        out_s2 += l2
        
    return scores, flags

# --- MAIN LOGIC ---

def run_suite(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    
    # 1. DISTRIBUTION MODE
    if args.n:
        print(f"\n=== LÄGE 1: FÖRDELNINGSANALYS ({args.n:,} sim) ===")
        print("Startar simulering...")
        
        total_counts = args.n
        chunk_size = min(total_counts, 200_000)
        processed = 0
        
        # Aggregatorer
        agg_s0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        agg_s1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        agg_s2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
        
        score_bins = {
            "all": np.zeros(376, dtype=np.int64), # Maxscore 375
            "yatzy": np.zeros(376, dtype=np.int64),
            "bonus": np.zeros(376, dtype=np.int64),
            "yatzy_bonus": np.zeros(376, dtype=np.int64)
        }
        
        start_t = time.time()
        
        while processed < total_counts:
            current_batch = min(chunk_size, total_counts - processed)
            
            # Kör seriell chunk (snabb nog för ~100k/sek per kärna, vi kör single-core stats för enkelhetens skull)
            # Vill man ha multicore här krävs map/reduce i Python. Numba serial är ca 3-4ms per spel? Nej, mycket snabbare.
            # Numba C++ speed är ca 1-2 mikrosekunder per spel. 200k games tar < 1 sek.
            
            t_chunk_start = time.time()
            # Använd seriell kernel för att få stats korrekta
            seed = np.random.randint(0, 2**30)
            sc, fl = _sim_chunk_serial(current_batch, seed, agg_s0, agg_s1, agg_s2)
            
            # Binning
            for s, f in zip(sc, fl):
                score_bins["all"][s] += 1
                is_bon = (f & 1)
                is_ytz = (f & 2)
                if is_ytz: score_bins["yatzy"][s] += 1
                if is_bon: score_bins["bonus"][s] += 1
                if is_ytz and is_bon: score_bins["yatzy_bonus"][s] += 1
            
            processed += current_batch
            
            # ETA Calc
            elapsed = time.time() - start_t
            rate = processed / elapsed
            eta = (total_counts - processed) / rate if rate > 0 else 0
            print(f"\rFramsteg: {processed/total_counts*100:.1f}% | {rate:,.0f} spel/s | ETA: {eta:.0f}s", end="")
        
        print("\nSimulering klar. Genererar filer...")
        
        # CSV 1: Distribution
        with open(out_dir / f"dist_scores_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Score", "Count_All", "Count_Yatzy", "Count_Bonus", "Count_YatzyBonus"])
            for s in range(376):
                if score_bins["all"][s] > 0:
                    w.writerow([s, score_bins["all"][s], score_bins["yatzy"][s], score_bins["bonus"][s], score_bins["yatzy_bonus"][s]])
                    
        # CSV 2: Category Stats
        with open(out_dir / f"dist_categories_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Roll1_Hits", "Roll1_Prob", "Roll2_Hits", "Roll2_Prob", "Roll3_Hits", "Roll3_Prob"])
            tot_r = total_counts * NUM_CATEGORIES
            for i, cat in enumerate(CATEGORY_NAMES):
                w.writerow([
                    cat,
                    agg_s0[i], agg_s0[i]/tot_r,
                    agg_s1[i], agg_s1[i]/tot_r,
                    agg_s2[i], agg_s2[i]/tot_r
                ])
                
        # Metadata
        elapsed_tot = time.time() - start_t
        mean_score = float(np.average(np.arange(376), weights=score_bins["all"]))
        
        meta = {
            "mode": "distribution",
            "count": total_counts,
            "elapsed_sec": elapsed_tot,
            "mean_score": mean_score,
            "system": platform.uname()._asdict()
        }
        with open(out_dir / f"meta_dist_{ts}.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"Klar! Data sparad i {out_dir}")

    # 2. DEVIATION STUDY MODE
    if args.study:
        steps = [int(x) for x in args.study.split(",")]
        reps = args.reps
        print(f"\n=== LÄGE 2: AVVIKELSSEANALYS ({len(steps)} steg, {reps} reps/steg) ===")
        
        results = []
        
        start_t_study = time.time()
        total_ops = sum(steps) * reps
        ops_done = 0
        
        for step in steps:
            for r in range(reps):
                # Kör en liten batch
                s0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
                s1 = np.zeros(NUM_CATEGORIES, dtype=np.int64) # Används ej men krävs av func
                s2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
                
                seed = np.random.randint(0, 2**30)
                _sim_chunk_serial(step, seed, s0, s1, s2)
                
                # Analysera Roll 0
                total_rolls = step * NUM_CATEGORIES
                obs_probs = s0 / total_rolls
                abs_devs = np.abs(obs_probs - EXPECTED_PROBS) * 100 # I procentenheter
                
                for i, cat in enumerate(CATEGORY_NAMES):
                    results.append({
                        "Simulations": step,
                        "Repetition": r+1,
                        "Category": cat,
                        "Expected_Pct": EXPECTED_PROBS[i] * 100,
                        "Observed_Pct": obs_probs[i] * 100,
                        "Abs_Deviation_Pct": abs_devs[i]
                    })
                
                ops_done += step
                elapsed = time.time() - start_t_study
                rate = ops_done / elapsed if elapsed > 0 else 0
                eta = (total_ops - ops_done) / rate if rate > 0 else 0
                print(f"\rSteg {step} (Rep {r+1}/{reps}) | Totalt: {ops_done/total_ops*100:.1f}% | ETA: {eta:.0f}s", end="")

        print("\nAnalys klar. Sparar data...")
        
        # CSV 3: Deviation
        with open(out_dir / f"study_deviation_{ts}.csv", "w", newline="") as f:
            fields = ["Simulations", "Repetition", "Category", "Expected_Pct", "Observed_Pct", "Abs_Deviation_Pct"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)

        # Metadata Study
        meta_study = {
            "mode": "deviation_study",
            "steps": steps,
            "reps": reps,
            "elapsed_sec": time.time() - start_t_study,
            "system": platform.uname()._asdict()
        }
        with open(out_dir / f"meta_study_{ts}.json", "w") as f:
            json.dump(meta_study, f, indent=2)

        print(f"Studie sparad i {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yatzy Suite: High-performance simulation & analysis")
    parser.add_argument("--n", type=int, help="Antal simuleringar för fördelningsanalys (t.ex. 1000000)")
    parser.add_argument("--study", type=str, help="Kommaseparerad lista på antal simuleringssteg för avvikelseanalys (t.ex. '1000,10000,100000')")
    parser.add_argument("--reps", type=int, default=5, help="Antal repetitioner per steg i study-mode (default: 5)")
    parser.add_argument("--output", type=str, default="results", help="Mapp för utdata")
    
    args = parser.parse_args()
    
    if not args.n and not args.study:
        parser.print_help()
    else:
        run_suite(args)