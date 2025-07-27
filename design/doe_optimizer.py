#!/usr/bin/env python
from __future__ import annotations
import itertools
from dataclasses import dataclass

ROWS = list("ABCDEFGH"); COLS = list(range(1,13))
GENO = ["WT","ΔcysH"]; BCD = ["low","med","high"]
IPTG = [0.0,0.05,0.1,0.5]; TIME = [4,8,18]

@dataclass(frozen=True)
class Cond:
    genotype: str; bcd: str; iptg: float; time: int

def score(c: Cond) -> float:
    s = 0.0
    if c.genotype == "ΔcysH": s += 2.0
    if c.bcd == "med": s += 1.0
    if c.time == 18: s += 1.0
    if 0.04 <= c.iptg <= 0.12: s += 0.8
    if c.iptg in (0.0, 0.5): s -= 0.2
    return s

def all_conds():
    for g,b,i,t in itertools.product(GENO, BCD, IPTG, TIME):
        yield Cond(g,b,i,t)

def top(n=18):
    xs = list(all_conds()); xs.sort(key=score, reverse=True); return xs[:n]

def plate_layout(conds):
    plate = [["" for _ in COLS] for _ in ROWS]
    for r in range(8): plate[r][0]  = f"STD{r+1}"
    for r in range(8): plate[r][11] = f"BLK{r+1}"
    slots = [(r,c) for c in range(1,11) for r in range(8)]
    for cnd,(r,c) in zip(conds, slots):
        plate[r][c] = f"{cnd.genotype}|{cnd.bcd}|{cnd.iptg}mM|{cnd.time}h"
    return plate

def print_plate(plate):
    print("    " + " ".join(f"{c:>4}" for c in COLS))
    for r,row in enumerate(ROWS):
        cells = [f"{row:>2} "] + [f"{(plate[r][c-1] or '.'):>4}" for c in COLS]
        print(" ".join(cells))

def main():
    tops = top(18)
    print("Top 18 conditions:")
    for i,c in enumerate(tops,1):
        print(f"{i:2d}. {c.genotype}, BCD={c.bcd}, IPTG={c.iptg} mM, t={c.time} h (score={score(c):.2f})")
    print("\n96-well draft (STD col1, BLK col12):")
    print_plate(plate_layout(tops))

if __name__ == "__main__":
    main()
