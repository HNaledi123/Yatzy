import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

## --- CONFIGURATION ---
debug = False

## --- HELPERS ---

def KastaTärningar(antal):
    return np.random.randint(1,7,antal).tolist()

def UtvärderaÖvreDel(tärningar, sökning):
    värde = 0
    for n in tärningar:
        if n == sökning:
            värde += sökning
    return värde

def UtvärderaTvåPar(tärningar):
    värde = 0
    required = 0
    counts = Counter(tärningar)
    for val, count in counts.items():
        if count == 2 or count == 3:
            required += 1
            värde += val * 2
    if required < 2:
        värde = 0
    return värde

def UtvärderaParTrissFyrtalYatzy(tärningar, krav):
    värde = 0
    counts = Counter(tärningar)
    for val, count in counts.items():
        if count >= krav:
            if krav == 5:
                värde = 50
            else:
                värde = max(värde, val * krav)
    return värde

def UtvärderaStege(tärningar,storlek):
    värde = 0
    if sorted(tärningar) == list(range(1+storlek, 6+storlek)):
        värde = 15 + storlek * 5
    return värde

def UtvärderaKåk(tärningar):
    värde = 0
    har_par = False
    har_triss = False
    counts = Counter(tärningar)
    for val, count in counts.items():
        if count == 2:
            har_par = True
            värde += val * 2
        if count == 3:
            har_triss = True
            värde += val * 3
    if not (har_par and har_triss):
        värde = 0
    return värde

def UtvärderaChans(tärningar):
    return sum(tärningar)

def UtvärderaKast(tärningar, tillåtna):
    värde = -1
    vald = ""

    for ö in range(6, 0, -1):
        utvärdering = UtvärderaÖvreDel(tärningar,ö)
        if utvärdering > värde and ("ÖvreDel" + str(ö)) in tillåtna:
            värde = utvärdering
            vald = "ÖvreDel" + str(ö)

    for p in range(5, 1, -1):
        utvärdering = UtvärderaParTrissFyrtalYatzy(tärningar,p)
        if utvärdering > värde and ("Lika" + str(p)) in tillåtna:
            värde = utvärdering
            vald = "Lika" + str(p)

    utvärdering = UtvärderaTvåPar(tärningar)
    if utvärdering > värde and "TvåPar" in tillåtna:
        värde = utvärdering
        vald = "TvåPar"

    for s in range(1, -1, -1):
        utvärdering = UtvärderaStege(tärningar,s)
        if utvärdering > värde and ("Stege" + str(s)) in tillåtna:
            värde = utvärdering
            vald = "Stege" + str(s)

    utvärdering = UtvärderaKåk(tärningar)
    if utvärdering > värde and "Kåk" in tillåtna:
        värde = utvärdering
        vald = "Kåk"

    utvärdering = UtvärderaChans(tärningar)
    if utvärdering > värde and "Chans" in tillåtna:
        värde = utvärdering
        vald = "Chans"

    return värde, vald

def KravutvärderaKast(tärningar):
    uppfyller = []

    for ö in range(6, 0, -1):
        utvärdering = UtvärderaÖvreDel(tärningar,ö)
        if utvärdering > 0:
            uppfyller.append("ÖvreDel" + str(ö))

    for p in range(5, 1, -1):
        utvärdering = UtvärderaParTrissFyrtalYatzy(tärningar,p)
        if utvärdering > 0:
            uppfyller.append("Lika" + str(p))

    utvärdering = UtvärderaTvåPar(tärningar)
    if utvärdering > 0:
        uppfyller.append("TvåPar")

    for s in range(1, -1, -1):
        utvärdering = UtvärderaStege(tärningar,s)
        if utvärdering > 0:
            uppfyller.append("Stege" + str(s))

    utvärdering = UtvärderaKåk(tärningar)
    if utvärdering > 0:
        uppfyller.append("Kåk")

    utvärdering = UtvärderaChans(tärningar)
    if utvärdering > 0:
        uppfyller.append("Chans")

    return uppfyller

def UtvärderaBonus(poäng):
    värde = 0
    if poäng >= 63:
        värde = 50
    if debug:
        print("Bonus | " + str(värde))
    return värde

# OMAKSTSSTRATEGI:
# Vid varje kast behåll flest lika tärningar, kasta om övriga.

def Omkast(tärningar):
    nyatärningar = []
    maxcount = 0
    maxval = 0
    counts = Counter(tärningar)
    for val, count in counts.items():
        if count > maxcount or (count == maxcount and val > maxval):
            maxval = val
            maxcount = count
    i = maxcount
    while i > 0:
        nyatärningar.append(int(maxval))
        i -= 1
    omkast = KastaTärningar(5-maxcount)
    for n in omkast:
        nyatärningar.append(n)
    return nyatärningar

## --- MAIN ---

def SpelaYatzy():
    tillåtnarutor = ["ÖvreDel1","ÖvreDel2","ÖvreDel3","ÖvreDel4","ÖvreDel5","ÖvreDel6","Lika2","Lika3","Lika4","Lika5","TvåPar","Stege0","Stege1","Kåk","Chans"]
    poäng = 0
    övrepoäng = 0
    kast = 1
    statistik = {}
    for r in tillåtnarutor:
        statistik.update({r:0})
    statistikomk1 = statistik.copy()
    statistikomk2 = statistik.copy()
    while kast < 16:
        tärningar = KastaTärningar(5)
        tärningaromk1 = Omkast(tärningar)
        tärningaromk2 = Omkast(tärningaromk1)
        utvärdering = UtvärderaKast(tärningaromk2, tillåtnarutor)
        kravkontrollutvärdering0 = KravutvärderaKast(tärningar)
        kravkontrollutvärdering1 = KravutvärderaKast(tärningaromk1)
        kravkontrollutvärdering2 = KravutvärderaKast(tärningaromk2)
        if debug:
            print("Uppfyllda Krav 0 (Kast " + str(kast) + ") | " + str(kravkontrollutvärdering0))
            print("Uppfyllda Krav 1 (Kast " + str(kast) + ") | " + str(kravkontrollutvärdering1))
            print("Uppfyllda Krav 2 (Kast " + str(kast) + ") | " + str(kravkontrollutvärdering2))
            print("Kast " + str(kast) + " | " + utvärdering[1] + " | " + str(utvärdering[0]))
        for k,s in statistik.items():
            if k in kravkontrollutvärdering0:
                s += 1
                statistik.update({k:s})
        for k,s in statistikomk1.items():
            if k in kravkontrollutvärdering1:
                s += 1
                statistikomk1.update({k:s})
        for k,s in statistikomk2.items():
            if k in kravkontrollutvärdering2:
                s += 1
                statistikomk2.update({k:s})

        poäng += utvärdering[0]
        tillåtnarutor.remove(utvärdering[1])
        if "ÖvreDel" in utvärdering[1]:
            övrepoäng += utvärdering[0]
        kast += 1
    bonus = UtvärderaBonus(övrepoäng)
    poäng += bonus
    return poäng, bonus > 0, statistik, statistikomk1, statistikomk2

def SimuleraRundor(antal):
    tillåtnarutor = ["ÖvreDel1","ÖvreDel2","ÖvreDel3","ÖvreDel4","ÖvreDel5","ÖvreDel6","Lika2","Lika3","Lika4","Lika5","TvåPar","Stege0","Stege1","Kåk","Chans"]
    start = time.time()
    resultat = []
    bonusar = 0
    storstats = {}
    storstats1 = {}
    storstats2 = {}
    for r in tillåtnarutor:
        storstats.update({r:0})
    storstats2 = storstats.copy()
    storstats1 = storstats.copy()
    for _ in range(antal):
        poäng, fick_bonus, stats, stats1, stats2 = SpelaYatzy()
        resultat.append(poäng)
        if fick_bonus:
            bonusar += 1
        for r,v in stats.items():
            ny = storstats[r]+v
            storstats.update({r:ny})
        for r,v in stats1.items():
            ny = storstats1[r]+v
            storstats1.update({r:ny})
        for r,v in stats2.items():
            ny = storstats2[r]+v
            storstats2.update({r:ny})

    slut = time.time()
    tid_ms = (slut - start) * 1000
    
    # Statistik
    medel = np.mean(resultat)
    std = np.std(resultat)
    bonus_sannolikhet = bonusar / antal * 100


    # Utskrift
    print(f"\n--- RESULTAT ({antal} spel) ---")
    print(f"\nTid: {tid_ms}ms")

    # --- CSV EXPORT  PLOTTNING ---

    df = pd.DataFrame({
        "Kategori": tillåtnarutor,
        "Roll0_count": [storstats[r] for r in tillåtnarutor],
        "Roll1_count": [storstats1[r] for r in tillåtnarutor],
        "Roll2_count": [storstats2[r] for r in tillåtnarutor],
    })

    df["Roll0_%"] = df["Roll0_count"] / antal / 15 * 100
    df["Roll1_%"] = df["Roll1_count"] / antal / 15 * 100
    df["Roll2_%"] = df["Roll2_count"] / antal / 15 * 100

    # Append summary rows
    summary = pd.DataFrame([
        {"Kategori": "Medelpoäng", "Roll0_count": medel},
        {"Kategori": "Std", "Roll0_count": std},
        {"Kategori": "Bonus%", "Roll0_count": bonus_sannolikhet}
    ])
    df = pd.concat([df, summary], ignore_index=True)

    csv_name = f"yatzy_stats_{antal}_{time.time()}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Resultat exporterat till: {csv_name}")

    # Plot utvecklingen av sannolikheter per kategori (BAR CHART)
    plt.figure(figsize=(10,5))

    kategorier = df["Kategori"][:-3]
    x = np.arange(len(kategorier))
    width = 0.25  # bar width

    plt.bar(x - width, df["Roll0_%"][:-3], width, label="Första kastet")
    plt.bar(x, df["Roll1_%"][:-3], width, label="Omkast 1")
    plt.bar(x + width, df["Roll2_%"][:-3], width, label="Omkast 2")

    plt.title("Sannolikhet att uppfylla ruta (första kast till andra omkast)")
    plt.xlabel("Yatzy-ruta")
    plt.ylabel("Sannolikhet (%)")
    plt.xticks(x, kategorier, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.hist(resultat, bins=20, edgecolor='black', alpha=0.7) 
    plt.axvline(medel, color='red', linestyle='--', linewidth=2, label=f"Medel = {medel:.1f}") 
    plt.title(f"Fördelning av Yatzy-poäng ({antal} simuleringar)") 
    plt.xlabel("Poäng") 
    plt.ylabel("Antal spel") 
    plt.legend() 
    plt.grid(alpha=0.3) 
    plt.tight_layout()
    plt.show()

SimuleraRundor(100)