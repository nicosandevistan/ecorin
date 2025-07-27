#!/usr/bin/env bash
set -euo pipefail

echo "[1] Locate Miniforge/conda…"
CONDA_BIN=""
for p in \
  /Applications/Miniforge3/bin/conda \
  /opt/homebrew/Caskroom/miniforge/*/Miniforge3/bin/conda \
  "$HOME/miniforge3/bin/conda" \
  "$HOME/Miniforge3/bin/conda"
do
  if [ -x "$p" ]; then CONDA_BIN="$p"; break; fi
done
if [ -z "${CONDA_BIN}" ]; then
  echo "!! conda not found. Install with: brew install --cask miniforge" >&2
  exit 1
fi
echo "[conda] $CONDA_BIN"

echo "[2] Ensure shell init and PATH"
# Add PATH for this process
export PATH="$(dirname "$CONDA_BIN"):$PATH"
# Initialize zsh for future sessions (safe to repeat)
"$CONDA_BIN" init zsh >/dev/null || true

# Load conda into this shell
if ! command -v conda >/dev/null 2>&1; then
  # Try the hook explicitly for zsh and bash
  eval "$("$CONDA_BIN" shell.zsh hook)" 2>/dev/null || eval "$("$CONDA_BIN" shell.bash hook)"
fi
conda --version

echo "[3] Create folders"
mkdir -p data/standards data/plate_raw data/metadata models design assays figs docs

echo "[4] Write env.yml"
cat > env.yml <<'YML'
name: ecorin
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scipy
  - matplotlib
  - sympy
  - pymc
  - arviz
  - numba
  - salib
  - jax
  - jaxlib
  - ipykernel
  - jupyterlab
  - black
  - flake8
  - pip
  - tqdm
  - xarray
  - pip:
      - nloed
      - logomaker
      - python-dotenv
YML

echo "[5] Create code & docs"
# Analysis plan
cat > docs/Analysis_Plan.md <<'MD'
# ECORIN Phase 0 — Analysis Plan
## Objective
Test whether **Δ*cysH*** (or CRISPRi) + **NDST2** increases **N-sulfation (HNS)** via higher **PAPS**.

## Hypotheses
H1: Removing *cysH* sink increases PAPS → higher NDST2 throughput.  
H2: Δ*cysH* + NDST2 > WT + NDST2 in HNS.  
H0: No difference.

## Endpoints
Primary: HNS (SU) per OD/biomass.  
Secondary: DMMB ΔA525–595; Heparin Red quench.  
QC: R², LoD/LoQ, spike recovery, plate CV%.

## ODE species
SO4, APS, PAPS, HS, HNS.

## Parameters (arb.)
k_uptake, k_aps, k_paps, vmax_cysh, k_paps_use, kcat_ndst2, Km_HS, paps_multiplier.  
WT baseline; Δ*cysH*: **vmax_cysh=0**, **paps_multiplier=10** (Phase 0 assumption).

## Inference
PyMC: two means (μ_WT, μ_KO), shared σ; NUTS. Save NetCDF, print summary.

## Sensitivity
Sobol first/total indices on final HNS(t_end) — plan for Phase 1.

## DoE
Factors: genotype {WT, Δ*cysH*}; NDST2 BCD {low, med, high}; IPTG {0, 0.05, 0.1, 0.5 mM}; time {4, 8, 18 h}.  
Heuristic prefers Δ*cysH*, med IPTG, 18 h. Future: D-optimal using Fisher information.

## Assay calibration
**DMMB pH 1.5:** ΔA = A525–A595; matrix-matched standards; LoD = 3.3·σ_blank/slope; LoQ = 10·σ_blank/slope; spikes.  
**Heparin Red:** 0–5 µg/mL; kit ex/em; background quench controls; LoD/LoQ; spikes.

## Risks → mitigations
Matrix effects → matrix-matched + spikes.  
Low range → dual assay + dilution.  
Burden/toxicity → growth controls, IPTG titration, BCD.  
Identifiability → sensitivity, informative priors, perturbations.  
Batch effects → randomization, triplicates, internal standards.

## Deliverables
ODE CLI (WT vs Δ*cysH* HNS), PyMC smoke, DoE selector + 96-well draft, SOP stubs, reproducible env.
MD

# paps_odes.py
cat > models/paps_odes.py <<'PY'
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
    if p is None: p = Params()
    g = genotype.lower()
    if g in {"delta_cysh", "dcysh", "ko", "ΔcysH".lower()}:
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
    if not sol.success: raise RuntimeError(sol.message)
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
    fold = (hns_ko / hns_wt) if hns_wt > 0 else np.inf
    print(f"  KO / WT: {fold:.2f}×")

if __name__ == "__main__":
    main()
PY
chmod +x models/paps_odes.py

# inference_pymc.py
cat > models/inference_pymc.py <<'PY'
#!/usr/bin/env python
from __future__ import annotations
import argparse, os, numpy as np, pymc as pm, arviz as az

def make_dummy(seed=42):
    rng = np.random.default_rng(seed)
    return {"wt": rng.normal(2.5, 0.6, 8), "ko": rng.normal(4.0, 0.6, 8)}

def fit(data, draws, tune, chains):
    y = np.concatenate([data["wt"], data["ko"]])
    group = np.array([0]*len(data["wt"]) + [1]*len(data["ko"]))
    with pm.Model():
        mu_wt = pm.Normal("mu_wt", 0, 10)
        mu_ko = pm.Normal("mu_ko", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        mu = pm.math.switch(pm.math.eq(group, 0), mu_wt, mu_ko)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=0.9, progressbar=True)
    return idata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--out", default="figs/inference_su_posterior.nc")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    draws, tune, chains = (250,250,2) if args.smoke_test else (1500,1500,4)
    idata = fit(make_dummy(), draws, tune, chains)
    az.to_netcdf(idata, args.out)
    print(f"Saved posterior to: {args.out}")
    print(az.summary(idata, var_names=["mu_wt","mu_ko","sigma"], round_to=3).to_string())

if __name__ == "__main__":
    main()
PY
chmod +x models/inference_pymc.py

# doe_optimizer.py
cat > design/doe_optimizer.py <<'PY'
#!/usr/bin/env python
from __future__ import annotations
import itertools
from dataclasses import dataclass
ROWS = list("ABCDEFGH"); COLS = list(range(1,13))
GENO = ["WT","ΔcysH"]; BCD = ["low","med","high"]
IPTG = [0.0,0.05,0.1,0.5]; TIME = [4,8,18]

@dataclass(frozen=True)
class Cond: genotype: str; bcd: str; iptg: float; time: int

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
PY
chmod +x design/doe_optimizer.py

# SOPs
cat > docs/SOP_DMMB_pH1p5.md <<'MD'
# SOP — DMMB pH 1.5 (ΔA525–595)
Purpose: quantify sulfated GAGs with ΔA = A525 − A595. Matrix-matched standards; spike recoveries; LoD = 3.3·σ_blank/slope; LoQ = 10·σ_blank/slope.
MD
cat > docs/SOP_HeparinRed.md <<'MD'
# SOP — Heparin Red (0–5 µg/mL)
Quench assay; kit ex/em (~570/610 nm); black plates; matrix-matched standards; background quench controls; LoD/LoQ; spike recovery.
MD
cat > docs/SOP_PlateLayout.md <<'MD'
# SOP — Plate Layout & Randomization
Col1 standards (STD1–STD8), Col12 blanks (BLK1–BLK8). Triplicates. Controls: no NDST2, 0 mM IPTG. Randomize across inner wells; balance genotypes.
MD

# Runner scripts
cat > docs/run_phase0_smoketests.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
eval "$(conda shell.zsh hook)" || eval "$(conda shell.bash hook)"
conda activate ecorin
python models/paps_odes.py --t_end 18 --ndst2 1.0
python models/inference_pymc.py --smoke-test --out figs/inference_su_posterior.nc
python design/doe_optimizer.py
echo "[done] Phase 0 smoke tests complete."
SH
chmod +x docs/run_phase0_smoketests.sh

cat > docs/git_commit_push.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
git add -A
git commit -m "Phase 0 baseline: env, plan, ODE, PyMC, DoE, SOPs, runner"
git push origin main
SH
chmod +x docs/git_commit_push.sh

echo "[6] Create/activate env if needed"
if ! conda env list | grep -q "ecorin"; then
  conda env create -f env.yml
fi
eval "$(conda shell.zsh hook)" || eval "$(conda shell.bash hook)"
conda activate ecorin

echo "[7] Run smoke tests"
bash docs/run_phase0_smoketests.sh

echo "[8] Verify artifacts"
test -s figs/inference_su_posterior.nc && echo "OK: NetCDF present"

echo "[9] Done. To commit: bash docs/git_commit_push.sh"
