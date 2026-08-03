"""
bsadf.py

Core PSY (Phillips, Shi & Yu, 2015, IER) recursive right-tailed unit root
testing procedure: BSADF statistic sequence and its supremum, GSADF.

The test is run on a single series y_t (here: log price-to-dividend ratio).
For every end point r2 (expanding through the sample) the ADF regression

    Delta y_t = alpha + beta * y_{t-1} + sum_{i=1..k} gamma_i * Delta y_{t-i} + e_t

is estimated on every window [r1, r2] with r1 <= r2 - w_min, and BSADF(r2) is
the largest resulting t-statistic on beta. GSADF is the largest BSADF value
over the whole sample. A right-tailed test (beta > 0, i.e. mildly explosive)
is used because bubbles imply locally explosive, not merely non-stationary,
behaviour.

The inner loops are numba-jitted because the recursive procedure evaluates
O(T^2) ADF regressions for the real data, and O(B * T^2) for each set of
critical values (B = number of Monte Carlo / bootstrap replications).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _adf_tstat(y_window: np.ndarray, k: int) -> float:
    """ADF t-stat on beta (coefficient of y_{t-1}) for one window, lag order k."""
    n = y_window.shape[0]
    dy = np.empty(n - 1)
    for i in range(n - 1):
        dy[i] = y_window[i + 1] - y_window[i]

    # usable rows of the regression: t = k+1 .. n-1 (0-indexed on dy), i.e. n-1-k rows
    n_obs = n - 1 - k
    n_reg = 2 + k  # const, y_{t-1}, k lagged differences
    if n_obs <= n_reg:
        return np.nan

    X = np.empty((n_obs, n_reg))
    yv = np.empty(n_obs)
    for row in range(n_obs):
        t = row + k  # index into dy for Delta y_t (0-indexed), t>=k
        yv[row] = dy[t]
        X[row, 0] = 1.0
        X[row, 1] = y_window[t]  # y_{t-1} in levels (index t in y_window == t-1 lag of dy[t])
        for i in range(k):
            X[row, 2 + i] = dy[t - 1 - i]

    XtX = X.T @ X
    Xty = X.T @ yv
    beta = np.linalg.solve(XtX, Xty)
    resid = yv - X @ beta
    dof = n_obs - n_reg
    if dof <= 0:
        return np.nan
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(XtX)
    se_beta1 = np.sqrt(sigma2 * XtX_inv[1, 1])
    if se_beta1 <= 0.0:
        return np.nan
    return beta[1] / se_beta1


@njit(cache=True)
def bsadf_sequence(y: np.ndarray, w_min: int, k: int) -> np.ndarray:
    """
    Recursive BSADF sequence for one series y (length T).

    Returns an array of length T; entries before w_min-1 are NaN (undefined,
    not enough data yet for even the smallest window).
    """
    T = y.shape[0]
    bsadf = np.full(T, np.nan)
    for r2 in range(w_min - 1, T):
        best = -1.0e300
        # window [r1, r2] must contain at least w_min points -> r1 <= r2 - w_min + 1
        for r1 in range(0, r2 - w_min + 2):
            stat = _adf_tstat(y[r1 : r2 + 1], k)
            if stat > best:
                best = stat
        bsadf[r2] = best
    return bsadf


@njit(cache=True)
def gsadf_stat(y: np.ndarray, w_min: int, k: int) -> float:
    seq = bsadf_sequence(y, w_min, k)
    best = -1.0e300
    for v in seq:
        if not np.isnan(v) and v > best:
            best = v
    return best


@njit(cache=True)
def bsadf_sequence_k0(y: np.ndarray, w_min: int) -> np.ndarray:
    """
    Fast O(T^2) specialisation of bsadf_sequence for lag order k=0, i.e. the
    plain Dickey-Fuller regression Delta y_t = alpha + beta*y_{t-1} + e_t.

    For fixed r2, as r1 decreases the window only grows by one point on the
    left, so the regression's sufficient statistics (sums of y_{t-1}, its
    square, Delta y_t, their cross product, and Delta y_t^2) can be updated
    in O(1) per step instead of rebuilding the whole window, cutting the
    total cost from O(T^3) to O(T^2). Mathematically identical to calling
    bsadf_sequence(y, w_min, 0); this is a performance-only specialisation.
    """
    T = y.shape[0]
    bsadf = np.full(T, np.nan)

    for r2 in range(w_min - 1, T):
        # smallest window: r1 = r2 - w_min + 1, size w_min (w_min-1 regression rows)
        r1 = r2 - w_min + 1
        n = 0
        Sy = 0.0
        Syy = 0.0
        Sdy = 0.0
        Sydy = 0.0
        Sdydy = 0.0
        for t in range(r1, r2):
            ylag = y[t]
            dy = y[t + 1] - y[t]
            n += 1
            Sy += ylag
            Syy += ylag * ylag
            Sdy += dy
            Sydy += ylag * dy
            Sdydy += dy * dy

        best = -1.0e300
        while True:
            if n > 2:
                mean_y = Sy / n
                mean_dy = Sdy / n
                Sxx = Syy - n * mean_y * mean_y
                Sxy = Sydy - n * mean_y * mean_dy
                if Sxx > 0.0:
                    beta = Sxy / Sxx
                    alpha = mean_dy - beta * mean_y
                    sse = Sdydy - alpha * Sdy - beta * Sydy
                    dof = n - 2
                    if sse > 0.0 and dof > 0:
                        sigma2 = sse / dof
                        se_beta = np.sqrt(sigma2 / Sxx)
                        if se_beta > 0.0:
                            stat = beta / se_beta
                            if stat > best:
                                best = stat
            if r1 == 0:
                break
            # extend window left by one point: new pair (y[r1-1], y[r1]-y[r1-1])
            r1 -= 1
            ylag = y[r1]
            dy = y[r1 + 1] - y[r1]
            n += 1
            Sy += ylag
            Syy += ylag * ylag
            Sdy += dy
            Sydy += ylag * dy
            Sdydy += dy * dy

        bsadf[r2] = best
    return bsadf


@njit(cache=True, parallel=True)
def bsadf_sequences_batch_k0(paths: np.ndarray, w_min: int) -> np.ndarray:
    """Parallel batch version of bsadf_sequence_k0 across B simulated paths."""
    B, T = paths.shape
    out = np.empty((B, T))
    for b in prange(B):
        out[b, :] = bsadf_sequence_k0(paths[b, :], w_min)
    return out


@njit(cache=True, parallel=True)
def bsadf_sequences_batch(paths: np.ndarray, w_min: int, k: int) -> np.ndarray:
    """
    paths: (B, T) array of B simulated series of length T.
    Returns (B, T) array of BSADF sequences, computed in parallel across replications.
    """
    B, T = paths.shape
    out = np.empty((B, T))
    for b in prange(B):
        out[b, :] = bsadf_sequence(paths[b, :], w_min, k)
    return out


def min_window(T: int, r0: float = None) -> int:
    """PSY (2015) rule of thumb: r0 = 0.01 + 1.8 / sqrt(T)."""
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)
    return int(np.floor(r0 * T))


def select_lag_bic(y: np.ndarray, kmax: int = 8) -> int:
    """Choose augmentation lag order k in [0, kmax] minimizing BIC on the full sample."""
    n = len(y)
    dy = np.diff(y)
    best_k, best_bic = 0, np.inf
    for k in range(0, kmax + 1):
        n_obs = n - 1 - k
        n_reg = 2 + k
        if n_obs <= n_reg + 1:
            break
        X = np.ones((n_obs, n_reg))
        yv = np.empty(n_obs)
        for row in range(n_obs):
            t = row + k
            yv[row] = dy[t]
            X[row, 1] = y[t]
            for i in range(k):
                X[row, 2 + i] = dy[t - 1 - i]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        rss = resid @ resid
        sigma2 = rss / n_obs
        bic = np.log(sigma2) + n_reg * np.log(n_obs) / n_obs
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k
