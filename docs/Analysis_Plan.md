# ECORIN Phase 0 — Analysis Plan
Objective: show Δ*cysH* + NDST2 increases N-sulfation (HNS) via higher PAPS.

Hypotheses: H1 PAPS↑ → NDST2 throughput↑. H2 KO>WT. H0 no difference.
Primary: HNS (SU) per OD/biomass. Secondary: DMMB ΔA525–595; Heparin Red. QC: R², LoD/LoQ, recovery, CV%.
ODE species: SO4, APS, PAPS, HS, HNS. Params: k_uptake, k_aps, k_paps, vmax_cysh, k_paps_use, kcat_ndst2, Km_HS, paps_multiplier.
Genotype: WT; Δ*cysH*: vmax_cysh=0; paps_multiplier=10 (assumption).
Inference: PyMC two means (μ_WT, μ_KO) shared σ; NUTS; NetCDF.
Sensitivity: Sobol on final HNS(t_end) (next phase).
DoE: genotype {WT, Δ*cysH*}; BCD {low, med, high}; IPTG {0, 0.05, 0.1, 0.5 mM}; time {4, 8, 18 h}. Heuristic favors KO, med IPTG, 18 h. Future: D-optimal.
Assays: DMMB pH 1.5 (ΔA = A525–A595; matrix-matched; LoD=3.3σ/slope; LoQ=10σ/slope; spikes). Heparin Red 0–5 µg/mL; kit ex/em; background quench controls.
Deliverables: ODE CLI; PyMC smoke + NetCDF; DoE top 18 + 96-well draft; SOP stubs; env.yml.
