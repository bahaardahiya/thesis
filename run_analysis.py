"""
run_analysis.py

Main entry point. Loads the cleaned NDX weekly log price-dividend ratio,
computes the BSADF/GSADF statistic sequence, generates critical values
under both the Monte Carlo (iid N(0,1)) and wild bootstrap procedures, and
writes all results to output/tables/ for downstream plotting and write-up.

Usage: python3 -m src.run_analysis
"""

import json
import time

import numpy as np
import pandas as pd

from src.bsadf import bsadf_sequence_k0, min_window, select_lag_bic
from src.critical_values import (
    monte_carlo_critical_values,
    wild_bootstrap_critical_values,
)

DATA_PATH = "data/processed/ndx_weekly_pd_ratio.csv"
OUT_TABLES = "output/tables"
N_REPLICATIONS = 4999
SEED_MC = 20250803
SEED_WB = 20250804
QUANTILES = (0.90, 0.95, 0.99)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    y = df["log_pd_ratio"].to_numpy()
    T = len(y)
    dates = df["date"]

    k = select_lag_bic(y, kmax=8)
    w_min = min_window(T)
    print(f"T={T}, BIC lag k={k}, w_min={w_min} weeks ({w_min/52:.2f} years)")
    if k != 0:
        raise RuntimeError(
            f"BIC selected lag k={k}, but this pipeline's fast path assumes k=0. "
            "Re-run with the general bsadf_sequence()/bsadf_sequences_batch() instead."
        )

    t0 = time.time()
    bsadf_stat = bsadf_sequence_k0(y, w_min)
    gsadf_stat = float(np.nanmax(bsadf_stat))
    gsadf_date = dates.iloc[int(np.nanargmax(bsadf_stat))]
    print(f"BSADF sequence computed in {time.time()-t0:.3f}s")
    print(f"GSADF = {gsadf_stat:.4f} (peak at {gsadf_date.date()})")

    print(f"\nSimulating {N_REPLICATIONS} Monte Carlo (iid N(0,1)) paths...")
    t0 = time.time()
    mc = monte_carlo_critical_values(T, w_min, N_REPLICATIONS, SEED_MC, QUANTILES)
    print(f"  done in {time.time()-t0:.3f}s")

    print(f"Simulating {N_REPLICATIONS} wild bootstrap paths...")
    t0 = time.time()
    wb = wild_bootstrap_critical_values(y, w_min, N_REPLICATIONS, SEED_WB, QUANTILES)
    print(f"  done in {time.time()-t0:.3f}s")

    # ---- assemble results table ----
    out = pd.DataFrame({"date": dates, "log_pd_ratio": y, "bsadf_stat": bsadf_stat})
    for q in QUANTILES:
        out[f"cv_mc_{int(q*100)}"] = mc["sequence"][q]
        out[f"cv_wb_{int(q*100)}"] = wb["sequence"][q]
    out.to_csv(f"{OUT_TABLES}/bsadf_results.csv", index=False)

    summary = {
        "T": T,
        "lag_k": k,
        "w_min_weeks": w_min,
        "n_replications": N_REPLICATIONS,
        "seed_mc": SEED_MC,
        "seed_wb": SEED_WB,
        "gsadf_stat": gsadf_stat,
        "gsadf_peak_date": str(gsadf_date.date()),
        "gsadf_cv_mc": mc["gsadf"],
        "gsadf_cv_wb": wb["gsadf"],
    }
    with open(f"{OUT_TABLES}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== GSADF one-shot test ===")
    print(f"GSADF stat = {gsadf_stat:.4f}")
    for q in QUANTILES:
        print(f"  MC {int(q*100)}% CV = {mc['gsadf'][q]:.4f}  | WB {int(q*100)}% CV = {wb['gsadf'][q]:.4f}")

    print(f"\nWrote {OUT_TABLES}/bsadf_results.csv and {OUT_TABLES}/summary.json")


if __name__ == "__main__":
    main()
