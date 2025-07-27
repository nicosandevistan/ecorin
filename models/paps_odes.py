#!/usr/bin/env python
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass
class Params:
    k_uptake: float = 0.05
    k_aps: float = 0.03
    k_paps: float = 0.05
    vmax_cysh: float = 0.02
    k_paps_use: float = 0.02
    kcat_ndst2: float = 0.1
    Km_HS: float = 1.0
    paps_multiplier: float = 1.0

def rhs(t, y, p: Params, ndst2_level: float):
    SO4, APS, PAPS, HS, HNS = y
    v_uptake = p.k_uptake * SO4
    dSO4 = -v_uptake
    v_aps_form = p.k_aps * SO4
    v_paps_form = p.k_paps * APS * p.paps_multiplier
    v_cysh = min(p.vmax_cysh, APS)
    dAPS = v_aps_form - v_paps_form - v_cysh
    v_ndst2 = ndst2_level * p.kcat_ndst2 * (HS / (p.Km_HS + HS)) * PAPS
    dPAPS = v_paps_form - v_ndst2 - p.k_paps_use * PAPS
    dHS = -v_ndst2
    dHNS = v_ndst2
    return np.array([dSO4, dAPS, dPAPS, dHS, dHNS], dtype=float)

def simulate(genotype: str, ndst2_level: float, t_end: float, p: Params | None = None):
    if p is None:
        p = Params()
    g = genotype.lower()
    if g in {"delta_cysh","dcysh","ko","ΔcysH".lower(),"Δcysh"}:
        p = Params(k_uptake=p.k_uptake, k_aps=p.k_aps, k_paps=p.k_paps,
                   vmax_cysh=0.0, k_paps_use=p.k_paps_use,
                   kcat_ndst2=p.kcat_ndst2, Km_HS=p.Km_HS,
                   paps_multiplier=10.0)
    elif g not in {"wt","wildtype"}:
        raise ValueError("genotype must be WT or ΔcysH")
    y0 = np.array([10.0, 0.5, 0.1, 5.0, 0.0], dtype=float)
    t_eval = np.linspace(0.0, float(t_end), 201)
    sol = solve_ivp(lambda t, y: rhs(t, y, p, ndst2_level),
                    (0.0, float(t_end)), y0, t_eval=t_eval,
                    method="LSODA", atol=1e-8, rtol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y

def final_hns(genotype: str, ndst2: float, t_end: float = 18.0) -> float:
    _, y = simulate(genotype, ndst2, t_end)
    return float(y[4, -1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_end", type=float, default=18.0)
    ap.add_argument("--ndst2", type=float, default=1.0)
    a = ap.parse_args()
    hns_wt = final_hns("WT", a.ndst2, a.t_end)
    hns_ko = final_hns("ΔcysH", a.ndst2, a.t_end)
    print(f"Final HNS @ t={a.t_end} h, NDST2={a.ndst2}")
    print(f"  WT     : {hns_wt:.4f}")
    print(f"  ΔcysH  : {hns_ko:.4f}")
    fold = (hns_ko / hns_wt) if hns_wt > 0 else float('inf')
    print(f"  KO / WT: {fold:.2f}×")

if __name__ == "__main__":
    main()
