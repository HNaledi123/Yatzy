import argparse
import csv
import json
import math
import os
import time
import sys
import concurrent.futures
from pathlib import Path
import numpy as np

# --- CONFIGURATION & CONSTANTS ---

# Försök ladda Numba för prestanda
try:
    from numba import njit
    # Vi behöver inte prange här eftersom vi använder ThreadPoolExecutor för att hantera trådar
    # på en högre nivå för att hantera statistisk aggregering säkrare.
    NUMBA_AVAILABLE = True
except ImportError:
    print("Varning: Numba saknas. Simuleringen kommer gå mycket långsamt.")
    sys.exit(1)

CATEGORY_NAMES = [
    "Upper1", "Upper2", "Upper3", "Upper4", "Upper5", "Upper6",
    "OfAKind2", "OfAKind3", "OfAKind4", "OfAKind5",
    "TwoPairs", "SmallStraight", "LargeStraight", "FullHouse", "Chance"
]

CATEGORY_TRANSLATION = {
    "Upper1": "Ettor", "Upper2": "Tvåor", "Upper3": "Treor", 
    "Upper4": "Fyror", "Upper5": "Femmor", "Upper6": "Sexor",
    "OfAKind2": "Ett par", "OfAKind3": "Tretal", "OfAKind4": "Fyrtal", 
    "OfAKind5": "Yatzy", "TwoPairs": "Två par", 
    "SmallStraight": "Liten stege", "LargeStraight": "Stor stege", 
    "FullHouse": "Kåk", "Chance": "Chans"
}

NUM_CATEGORIES = len(CATEGORY_NAMES)
FACE_TO_UPPER_IDX = {f: f - 1 for f in range(1, 7)}
OF_A_KIND_TO_IDX = {2: 6, 3: 7, 4: 8, 5: 9}
YATZY_IDX = 9

EXPECTED_PROBS = np.array([
    0.598122, 0.598122, 0.598122, 0.598122, 0.598122, 0.598122,
    0.907407, 0.212963, 0.020062, 0.000772, 0.270062, 
    0.015432, 0.015432, 0.038580, 1.000000
])

ROLL_STATE_COUNT = 6 ** 5

# --- LOOKUP TABLE GENERATION (Preserved for Exact Behavior) ---

def _build_lookup_tables():
    from itertools import product
    scores = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.int16)
    sat_mask = np.zeros((ROLL_STATE_COUNT, NUM_CATEGORIES), dtype=np.uint8)
    keep_val = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    keep_cnt = np.zeros(ROLL_STATE_COUNT, dtype=np.int8)
    
    priority = [5, 4, 3, 2, 1, 0, 9, 8, 7, 6, 10, 12, 11, 13, 14] 

    for idx, dice in enumerate(product(range(1, 7), repeat=5)):
        d_arr = np.array(dice, dtype=np.int8)
        counts = np.bincount(d_arr, minlength=7)
        s_dice = np.sort(d_arr)
        d_sum = d_arr.sum()

        for f in range(1, 7):
            scores[idx, f-1] = counts[f] * f
            if counts[f] > 0: sat_mask[idx, f-1] = 1
        
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
        
        pairs = [f for f in range(1, 7) if counts[f] >= 2]
        if len(pairs) >= 2:
            scores[idx, 10] = pairs[-1]*2 + pairs[-2]*2
            sat_mask[idx, 10] = 1
            
        u_vals = np.unique(s_dice)
        if np.array_equal(u_vals, [1,2,3,4,5]): 
            scores[idx, 11] = 15
            sat_mask[idx, 11] = 1
        if np.array_equal(u_vals, [2,3,4,5,6]): 
            scores[idx, 12] = 20
            sat_mask[idx, 12] = 1
            
        if np.any(counts == 3) and np.any(counts == 2):
            scores[idx, 13] = (np.where(counts==3)[0][0]*3 + np.where(counts==2)[0][0]*2)
            sat_mask[idx, 13] = 1

        scores[idx, 14] = d_sum
        sat_mask[idx, 14] = 1

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

print("Bygger lookup-tabeller...", end="", flush=True)
TBL_SCORES, TBL_SAT, TBL_KEEP_VAL, TBL_KEEP_CNT, TBL_PRIO = _build_lookup_tables()
print("Klar.")

# --- OPTIMIZED NUMBA KERNEL ---

@njit(cache=True, nogil=True)
def _encode(dice):
    k = 0
    for i in range(5): 
        k = k * 6 + (dice[i] - 1)
    return k

@njit(cache=True, nogil=True)
def _ai_reroll(dice, roll_idx, out):
    cnt = TBL_KEEP_CNT[roll_idx]
    if cnt == 5:
        # Kopiera över direkt om vi behåller allt
        for i in range(5):
            out[i] = dice[i]
        return
        
    val = TBL_KEEP_VAL[roll_idx]
    # Sätt de tärningar vi sparar
    for i in range(cnt): 
        out[i] = val
    # Slå om resten
    for i in range(cnt, 5): 
        out[i] = np.random.randint(1, 7)

@njit(cache=True, nogil=True)
def _play_game_optimized(stats0, stats1, stats2, d0, d1, d2):
    """
    Kör ett enda spel.
    Tar in pre-allokerade arrayer (d0, d1, d2) för att undvika minnesallokering i loopen.
    """
    allowed = 0x7FFF 
    total = 0
    upper = 0
    got_yatzy = False
    
    # Loop unrolling optimization implicit via logic, but keep structure for stats
    for _ in range(NUM_CATEGORIES): 
        # Roll 1
        for i in range(5): 
            d0[i] = np.random.randint(1, 7)
        idx0 = _encode(d0)
        
        # Ackumulera statistik för Roll 1
        # (Direkt indexering i global array är snabbt i L1 cache eftersom tabellen är liten)
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
        
        # Välj kategori
        best_sc = -1
        best_id = -1
        scores = TBL_SCORES[idx2]
        
        # Priority loop (kort, 15 varv)
        for i in range(NUM_CATEGORIES):
            cat = TBL_PRIO[i]
            # Bitwise check om kategorin är ledig
            if (allowed >> cat) & 1:
                s = scores[cat]
                if s > best_sc:
                    best_sc = s
                    best_id = cat
        
        total += best_sc
        if best_id < 6: 
            upper += best_sc
        if best_id == YATZY_IDX and best_sc > 0: 
            got_yatzy = True
        
        # Markera kategori som tagen
        allowed &= ~(1 << best_id)
        
    bonus = 50 if upper >= 63 else 0
    return total + bonus, bonus > 0, got_yatzy

@njit(cache=True, nogil=True)
def _worker_sim_batch(count, seed):
    """
    Arbetarfunktion som kör en batch av spel.
    Körs med nogil=True vilket tillåter multitrådning från Python.
    """
    np.random.seed(seed)
    
    # Utdata-arrayer
    scores_out = np.empty(count, dtype=np.int16)
    flags_out = np.empty(count, dtype=np.int8)
    
    # Lokala statistik-ackumulatorer för denna tråd
    l_s0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    l_s1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    l_s2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    
    # Pre-allokera tärningsbuffertar för att undvika malloc i loopen
    d0 = np.empty(5, dtype=np.int8)
    d1 = np.empty(5, dtype=np.int8)
    d2 = np.empty(5, dtype=np.int8)
    
    for i in range(count):
        sc, bon, ytz = _play_game_optimized(l_s0, l_s1, l_s2, d0, d1, d2)
        
        scores_out[i] = sc
        
        # Flagga: Bit 0 = Bonus, Bit 1 = Yatzy
        f = 0
        if bon: f |= 1
        if ytz: f |= 2
        flags_out[i] = f
        
    return scores_out, flags_out, l_s0, l_s1, l_s2

# --- DRIVER LOGIC (MULTI-THREADED) ---

def run_simulation_parallel(total_count, batch_size=None):
    """
    Kör simuleringen parallellt över alla tillgängliga CPU-kärnor.
    """
    if batch_size is None:
        # Auto-tune: Försök dela upp i lagom stora bitar för att hålla alla kärnor upptagna
        # men inte så små att overhead blir stor.
        cpu_count = os.cpu_count() or 4
        # Dela upp totalen på ca 4 * cpu_count chunks för bra balans
        target_chunks = cpu_count * 4
        batch_size = max(1000, total_count // target_chunks)
        # Cap batch size för att inte blockera progress bar för länge
        batch_size = min(batch_size, 100_000)

    # Globala ackumulatorer
    agg_s0 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_s1 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    agg_s2 = np.zeros(NUM_CATEGORIES, dtype=np.int64)
    
    all_scores = []
    all_flags = []
    
    futures = []
    processed = 0
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Skapa jobb
        remaining = total_count
        while remaining > 0:
            count = min(batch_size, remaining)
            # Generera seed i Python (main thread) för att garantera unikhet
            seed = np.random.randint(0, 2**30)
            futures.append(executor.submit(_worker_sim_batch, count, seed))
            remaining -= count
            
        # Hämta resultat
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            res_scores, res_flags, r_s0, r_s1, r_s2 = f.result()
            
            # Aggregera statistik
            agg_s0 += r_s0
            agg_s1 += r_s1
            agg_s2 += r_s2
            
            # Spara rådata (kan optimeras med pre-allokering om minne är trångt, men list extend är snabbt nog här)
            all_scores.append(res_scores)
            all_flags.append(res_flags)
            
            completed += len(res_scores)
            
            # Progress bar
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total_count - completed) / rate if rate > 0 else 0
            print(f"\rSimulerar: {completed/total_count*100:.1f}% | {rate:,.0f} spel/s | ETA: {eta:.0f}s", end="")
            
    print() # Newline after progress
    
    # Slå ihop arrayer
    final_scores = np.concatenate(all_scores)
    final_flags = np.concatenate(all_flags)
    
    return final_scores, final_flags, agg_s0, agg_s1, agg_s2

# --- MAIN LOGIC ---

def run_suite(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    
    # 1. DISTRIBUTION MODE
    if args.n:
        print(f"\n=== LÄGE 1: FÖRDELNINGSANALYS (ULTRA) ({args.n:,} sim) ===")
        
        start_t = time.time()
        
        scores, flags, s0, s1, s2 = run_simulation_parallel(args.n)
        
        print("Bearbetar data och sparar filer...")
        
        # Binning logic (vektoriserad i numpy istället för loop för hastighet)
        score_bins = np.bincount(scores, minlength=376)
        
        # Binning av sub-kategorier
        # flags: 0=None, 1=Bonus, 2=Yatzy, 3=Both
        # Vi använder logical indexing
        mask_bonus = (flags & 1) > 0
        mask_yatzy = (flags & 2) > 0
        
        # Skapa 4 grupper
        # NoYatzy_NoBonus: ~Y & ~B
        # NoYatzy_YesBonus: ~Y & B
        # YesYatzy_NoBonus: Y & ~B
        # YesYatzy_YesBonus: Y & B
        
        bins_ny_nb = np.bincount(scores[~mask_yatzy & ~mask_bonus], minlength=376)
        bins_ny_yb = np.bincount(scores[~mask_yatzy & mask_bonus], minlength=376)
        bins_yy_nb = np.bincount(scores[mask_yatzy & ~mask_bonus], minlength=376)
        bins_yy_yb = np.bincount(scores[mask_yatzy & mask_bonus], minlength=376)
        
        # CSV 1: Distribution
        with open(out_dir / f"dist_scores_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Score", "Count_Total", "NoYatzy_NoBonus", "NoYatzy_YesBonus", "YesYatzy_NoBonus", "YesYatzy_YesBonus"])
            for s in range(376):
                if score_bins[s] > 0:
                    w.writerow([
                        s, 
                        score_bins[s], 
                        bins_ny_nb[s], 
                        bins_ny_yb[s], 
                        bins_yy_nb[s], 
                        bins_yy_yb[s]
                    ])

        # CSV 2: Category Stats
        tot_r = args.n * NUM_CATEGORIES
        with open(out_dir / f"dist_categories_{ts}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Roll1_Hits", "Roll1_Prob", "Roll2_Hits", "Roll2_Prob", "Roll3_Hits", "Roll3_Prob"])
            for i, cat in enumerate(CATEGORY_NAMES):
                w.writerow([
                    cat,
                    s0[i], s0[i]/tot_r,
                    s1[i], s1[i]/tot_r,
                    s2[i], s2[i]/tot_r
                ])
                
        elapsed_tot = time.time() - start_t
        mean_score = np.mean(scores)
        
        meta = {
            "mode": "distribution",
            "count": args.n,
            "elapsed_sec": elapsed_tot,
            "mean_score": float(mean_score),
            "performance": f"{args.n/elapsed_tot:.0f} games/sec"
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
        summary_data = {cat: {step: [] for step in steps} for cat in CATEGORY_NAMES}
        
        start_t_study = time.time()
        
        # För study mode kör vi mindre batcher ofta, men vi kan fortfarande använda parallellmotorn
        # för de större stegen.
        
        for step in steps:
            for r in range(reps):
                print(f"\rKör simulering: Steg {step}, Rep {r+1}/{reps}...", end="")
                
                # Kör parallellt (eller seriellt om steget är litet, hanteras av batch logic)
                # Vi ignorerar score/flags här, behöver bara s0
                _, _, s0, _, _ = run_simulation_parallel(step, batch_size=max(1000, step//os.cpu_count()))
                
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

        print("\nAnalys klar. Sparar data...")
        
        # CSV 3: Deviation Detail
        with open(out_dir / f"study_deviation_{ts}.csv", "w", newline="") as f:
            fields = ["Simulations", "Repetition", "Category", "Expected_Pct", "Observed_Pct", "Abs_Deviation_Pct"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)

        # CSV 4: Deviation Summary
        with open(out_dir / f"study_summary_{ts}.csv", "w", newline="") as f:
            headers = ["Kategori"] + [f"Sim_{s}" for s in steps]
            w = csv.writer(f)
            w.writerow(headers)
            
            for cat in CATEGORY_NAMES:
                swe_name = CATEGORY_TRANSLATION.get(cat, cat)
                row = [swe_name]
                for step in steps:
                    devs = summary_data[cat][step]
                    avg_dev = sum(devs) / len(devs) if devs else 0
                    row.append(f"{avg_dev:.4f}")
                w.writerow(row)

        meta_study = {
            "mode": "deviation_study",
            "steps": steps,
            "reps": reps,
            "elapsed_sec": time.time() - start_t_study
        }
        with open(out_dir / f"meta_study_{ts}.json", "w") as f:
            json.dump(meta_study, f, indent=2)

        print(f"Studie sparad i {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yatzy Suite: Ultra-High-Performance")
    parser.add_argument("--n", type=int, help="Antal simuleringar för fördelningsanalys (t.ex. 1000000)")
    parser.add_argument("--study", type=str, help="Kommaseparerad lista på antal simuleringssteg (t.ex. '1000,10000')")
    parser.add_argument("--reps", type=int, default=5, help="Antal repetitioner per steg i study-mode")
    parser.add_argument("--output", type=str, default="results", help="Mapp för utdata")
    
    args = parser.parse_args()
    
    if not args.n and not args.study:
        parser.print_help()
    else:
        run_suite(args)