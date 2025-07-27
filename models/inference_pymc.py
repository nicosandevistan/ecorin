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
