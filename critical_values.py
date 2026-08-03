"""
critical_values.py

Two ways of generating (finite-sample, right-tailed) critical value
sequences for the BSADF/GSADF statistics, both simulated under the null of
H0: y_t is a pure random walk (no bubble):

1. Monte Carlo (PSY 2015 standard approach): innovations are drawn iid
   N(0,1). This is scale-free (Brownian motion is self-similar), so only
   the sample size T and the minimum window w_min matter -- it does not use
   the actual data at all beyond its length.

2. Wild bootstrap (Phillips & Shi, 2020): innovations are the *actual*
   first differences of the observed log price-dividend ratio, each
   multiplied by an iid Rademacher sign (+-1 w.p. 1/2), then cumulated into
   a bootstrap path. This preserves the empirical heteroskedasticity /
   volatility-clustering pattern of the real series under the null, which
   iid-N(0,1) Monte Carlo cannot do. It is the theoretically appropriate
   choice when the data show conditional heteroskedasticity (as equity
   index returns typically do), because iid Monte Carlo critical values are
   otherwise size-distorted (typically too low, over-rejecting H0 / flagging
   spurious "bubbles" that are really just volatility clusters).

Both procedures compute the BSADF sequence on each simulated/bootstrapped
path (lag order fixed at k=0, matching the BIC choice on the real data —
see bsadf.select_lag_bic) and take empirical quantiles across replications,
at each r2, to form the critical value sequence.
"""

import numpy as np

from src.bsadf import bsadf_sequences_batch_k0

DEFAULT_QUANTILES = (0.90, 0.95, 0.99)


def simulate_mc_paths(T: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B iid-N(0,1) random walks of length T, y_0 = 0."""
    innovations = rng.standard_normal((B, T))
    innovations[:, 0] = 0.0
    return np.cumsum(innovations, axis=1)


def simulate_wb_paths(y: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """
    B wild-bootstrap random walks of length T built from the actual first
    differences of y, each week's shock independently sign-flipped
    (Rademacher multiplier) across replications.
    """
    T = len(y)
    e_hat = np.diff(y)  # length T-1, the real weekly log P/D changes
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, T - 1))
    boot_innov = signs * e_hat[np.newaxis, :]
    paths = np.empty((B, T))
    paths[:, 0] = 0.0
    paths[:, 1:] = np.cumsum(boot_innov, axis=1)
    return paths


def critical_value_sequences(
    bsadf_batch: np.ndarray, quantiles=DEFAULT_QUANTILES
) -> dict:
    """
    bsadf_batch: (B, T) array of simulated BSADF sequences (with NaN before
    w_min-1, matching bsadf.bsadf_sequence's convention).

    Returns dict: {quantile: (T,) array of critical values at each r2},
    plus 'gsadf': {quantile: scalar} critical values for the one-shot GSADF
    statistic (sup over r2 of each replication's BSADF sequence).
    """
    T = bsadf_batch.shape[1]
    cv_seq = {q: np.full(T, np.nan) for q in quantiles}
    valid_cols = ~np.all(np.isnan(bsadf_batch), axis=0)
    for q in quantiles:
        cv_seq[q][valid_cols] = np.nanquantile(bsadf_batch[:, valid_cols], q, axis=0)

    row_max = np.nanmax(bsadf_batch, axis=1)  # GSADF stat per replication
    cv_gsadf = {q: float(np.quantile(row_max, q)) for q in quantiles}

    return {"sequence": cv_seq, "gsadf": cv_gsadf}


def monte_carlo_critical_values(
    T: int, w_min: int, B: int, seed: int, quantiles=DEFAULT_QUANTILES
) -> dict:
    rng = np.random.default_rng(seed)
    paths = simulate_mc_paths(T, B, rng)
    batch = bsadf_sequences_batch_k0(paths, w_min)
    return critical_value_sequences(batch, quantiles)


def wild_bootstrap_critical_values(
    y: np.ndarray, w_min: int, B: int, seed: int, quantiles=DEFAULT_QUANTILES
) -> dict:
    rng = np.random.default_rng(seed)
    paths = simulate_wb_paths(y, B, rng)
    batch = bsadf_sequences_batch_k0(paths, w_min)
    return critical_value_sequences(batch, quantiles)
