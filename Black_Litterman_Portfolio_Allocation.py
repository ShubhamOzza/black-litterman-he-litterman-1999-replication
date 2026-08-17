"""
Black-Litterman Portfolio Allocation
====================================

Project I-8 of the Quant Master project series (Portfolio Construction domain).

This script implements the full Black-Litterman (BL) asset allocation workflow:

    1. Data layer: monthly total returns for a multi-asset universe of 8 to 12
       asset-class ETFs covering 2015 to 2025. Three input modes are supported:
           * synthetic : a clearly labeled, reproducible simulated dataset used
             for demonstration and testing when live market data is unavailable;
           * csv       : a user-supplied CSV of monthly returns or prices;
           * yfinance  : downloads monthly adjusted prices from Yahoo Finance.
    2. Equilibrium layer: implied equilibrium excess returns are recovered by
       inverse optimization, pi = delta * Sigma * w_mkt, where Sigma is a
       shrunk (Ledoit-Wolf style) covariance estimate and w_mkt is a vector of
       market-capitalization style weights.
    3. Views layer: the investor expresses K views P * mu = Q with an
       uncertainty matrix Omega proportional to P * Sigma * P^T and a scalar
       tau (default 0.05) scaling the uncertainty of the equilibrium prior.
    4. Posterior layer: the BL posterior expected return vector and covariance
       matrix are computed in closed form.
    5. Optimization layer: constrained mean-variance optimization with weights
       summing to one, long-only bounds, and per-asset caps (25% by default),
       plus a tracking-error-constrained variant relative to a benchmark and
       a constraint-cost table quantifying the utility given up at each
       constraint tier. The solver is cvxpy when available, with a scipy
       SLSQP fallback otherwise.
    6. Diagnostics layer:
           * naive mean-variance failure analysis (error maximization,
             concentration, instability under input perturbation);
           * unit tests of the BL limiting cases (zero-confidence views recover
             equilibrium exactly; infinite-confidence views satisfy P*mu = Q);
           * quarterly walk-forward comparison from 2020 onward against a
             60/40 stock/bond benchmark and an equal-weight portfolio;
           * tau and view-confidence sensitivity studies.
    7. Output layer: figures (PNG) and summary tables (CSV) written to an
       output directory, plus a printed run log.

IMPORTANT HONESTY NOTE
----------------------
The synthetic demo mode exists so the complete workflow can be executed and
validated without market data access. Synthetic results are NOT real market
evidence and must never be presented as such. All synthetic artifacts are
labeled accordingly in file names, figure titles, and console output.

Usage examples
--------------
    python Black_Litterman_Portfolio_Allocation.py --mode synthetic
    python Black_Litterman_Portfolio_Allocation.py --mode csv --csv my_returns.csv
    python Black_Litterman_Portfolio_Allocation.py --mode csv --csv my_prices.csv --prices
    python Black_Litterman_Portfolio_Allocation.py --mode yfinance
    python Black_Litterman_Portfolio_Allocation.py --selftest

Author: Quant Master project series
"""

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import warnings

# -----------------------------------------------------------------------------
# Third-party imports: numerical core
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Matplotlib is used in non-interactive ("Agg") mode so the script runs on
# headless machines (servers, CI runners) without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scipy provides the fallback quadratic-programming-free optimizer (SLSQP).
from scipy.optimize import minimize

# scikit-learn provides the Ledoit-Wolf shrinkage covariance estimator. If it
# is unavailable we fall back to a simple diagonal-shrinkage implementation
# written directly in this file (see shrink_covariance).
try:
    from sklearn.covariance import LedoitWolf

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SKLEARN_AVAILABLE = False

# cvxpy provides a disciplined convex programming interface for the
# constrained mean-variance problem. If unavailable we use scipy SLSQP.
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    CVXPY_AVAILABLE = False

# yfinance is only needed for the (optional) live download mode.
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    YFINANCE_AVAILABLE = False


# =============================================================================
# Global configuration
# =============================================================================

MONTHS_PER_YEAR = 12            # used to annualize monthly means and covariances
RISK_AVERSION = 2.5             # delta: average investor risk aversion for pi = delta*Sigma*w
TAU = 0.05                      # tau: uncertainty scale of the equilibrium prior
ASSET_CAP = 0.25                # per-asset long-only weight cap (project spec: max 25% per asset)
TE_BUDGET = 0.05                # annualized tracking-error budget for the TE-constrained variant
ESTIMATION_WINDOW = 60          # trailing months used at each walk-forward rebalance
REBALANCE_STEP = 3              # quarterly rebalancing (months between rebalances)
WALK_FORWARD_START = "2020-01-31"  # first out-of-sample month
RANDOM_SEED = 42                # fixed seed for full reproducibility

# Numerical tolerances used by the unit tests and sanity checks.
TOL_LIMITING_CASE = 1e-6        # tolerance for BL limiting-case identities
TOL_CONSTRAINT = 1e-6           # tolerance for optimizer constraint checks
TOL_PSD = -1e-10                # smallest acceptable covariance eigenvalue


# =============================================================================
# Asset universe definition
# =============================================================================
# Ten liquid asset-class ETFs proxy the global opportunity set. The tickers are
# used in yfinance mode; the names are used throughout the reporting layer.
# Synthetic mode calibrates each asset's expected return and volatility to be
# broadly plausible relative to long-run historical experience, without being
# tied to any specific sample path.

ASSET_UNIVERSE = [
    # (display name,          ticker,  synthetic ann. return, synthetic ann. vol)
    ("US Large Cap Equity",   "SPY",   0.080, 0.150),
    ("US Small Cap Equity",   "IWM",   0.085, 0.190),
    ("Developed ex-US Equity", "EFA",  0.060, 0.165),
    ("Emerging Markets Equity", "EEM", 0.070, 0.210),
    ("US Aggregate Bonds",    "AGG",   0.030, 0.045),
    ("Long US Treasuries",    "TLT",   0.035, 0.120),
    ("Investment Grade Credit", "LQD", 0.038, 0.060),
    ("High Yield Credit",     "HYG",   0.055, 0.090),
    ("Real Estate (REITs)",   "VNQ",   0.075, 0.180),
    ("Gold",                  "GLD",   0.045, 0.140),
]

# Market-capitalization style weights for the synthetic equilibrium portfolio.
# These are rough, plausible relative sizes of global investable wealth by
# asset class; they sum to exactly one and, deliberately, stay one point
# below ASSET_CAP (25%) at the largest position. This keeps the equilibrium
# portfolio itself strictly feasible under the project's own per-asset cap,
# which is what allows the forward optimizer to recover w_mkt exactly from
# pi = delta * Sigma * w_mkt (see the "Equilibrium optimal" column in the
# full-sample weights table) - an internal-consistency check that would
# silently break if the reference market weights violated the cap - while
# still leaving enough headroom for the tau and confidence sensitivity
# studies to show the largest position's weight move smoothly toward the
# cap rather than sitting pinned on it for every parameter value.
SYNTHETIC_MARKET_WEIGHTS = np.array(
    [0.24, 0.07, 0.15, 0.08, 0.17, 0.08, 0.08, 0.05, 0.05, 0.03]
)

# Synthetic monthly correlation matrix. Correlations are chosen to reflect
# well-known stylized facts: equities are strongly correlated with each other,
# long Treasuries hedge equity risk modestly, credit sits between equities and
# rates, and gold is close to uncorrelated. Correlations are calibrated so
# that the implied equilibrium risk premia pi = delta * Sigma * w_mkt are all
# positive and plausible. The matrix is validated (and, if ever needed,
# projected) to be positive semidefinite at runtime.
SYNTHETIC_CORR = np.array(
    [
        # SPY   IWM   EFA   EEM   AGG   TLT   LQD   HYG   VNQ   GLD
        [1.00, 0.85, 0.88, 0.78, 0.10, -0.10, 0.25, 0.62, 0.72, 0.05],
        [0.85, 1.00, 0.80, 0.72, 0.05, -0.10, 0.18, 0.60, 0.75, 0.02],
        [0.88, 0.80, 1.00, 0.82, 0.08, -0.10, 0.28, 0.60, 0.65, 0.08],
        [0.78, 0.72, 0.82, 1.00, 0.05, -0.10, 0.22, 0.58, 0.60, 0.12],
        [0.10, 0.05, 0.08, 0.05, 1.00, 0.65, 0.80, 0.30, 0.15, 0.20],
        [-0.10, -0.10, -0.10, -0.10, 0.65, 1.00, 0.40, -0.05, -0.05, 0.18],
        [0.25, 0.18, 0.28, 0.22, 0.80, 0.40, 1.00, 0.55, 0.30, 0.18],
        [0.62, 0.60, 0.60, 0.58, 0.30, -0.05, 0.55, 1.00, 0.55, 0.08],
        [0.72, 0.75, 0.65, 0.60, 0.15, -0.05, 0.30, 0.55, 1.00, 0.05],
        [0.05, 0.02, 0.08, 0.12, 0.20, 0.18, 0.18, 0.08, 0.05, 1.00],
    ]
)


# =============================================================================
# Section 1: Data layer
# =============================================================================

def nearest_psd_correlation(corr, tol=1e-10):
    """Project a symmetric matrix onto the cone of valid correlation matrices.

    The projection clips negative eigenvalues to zero (Higham-style spectral
    projection) and renormalizes the diagonal to one. In practice the hand-set
    SYNTHETIC_CORR above is already positive definite, so this is a safety net
    rather than an active transformation.

    Parameters
    ----------
    corr : (n, n) ndarray
        Symmetric candidate correlation matrix.
    tol : float
        Eigenvalues below this value are lifted to zero.

    Returns
    -------
    (n, n) ndarray
        Positive semidefinite correlation matrix with unit diagonal.
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, tol)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    return psd


def generate_synthetic_monthly_returns(seed=RANDOM_SEED):
    """Generate the clearly labeled synthetic monthly return dataset.

    Returns are drawn from a multivariate normal distribution calibrated to
    the per-asset expected returns, volatilities, and correlations defined in
    the configuration block. The sample spans January 2015 through December
    2025 (132 months), matching the project's 2015 to 2025 data target.

    This dataset is SYNTHETIC. It exists so the full pipeline can be executed,
    tested, and demonstrated without market data access. It must never be
    presented as real market evidence.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random generator, fixed for reproducibility.

    Returns
    -------
    pandas DataFrame
        Monthly simple returns with a month-end DatetimeIndex and one column
        per asset.
    """
    rng = np.random.default_rng(seed)
    names = [a[0] for a in ASSET_UNIVERSE]
    mu_ann = np.array([a[2] for a in ASSET_UNIVERSE])
    vol_ann = np.array([a[3] for a in ASSET_UNIVERSE])

    # Build the annualized covariance matrix Sigma = D * R * D from the
    # volatility vector D and the (PSD-validated) correlation matrix R.
    corr = nearest_psd_correlation(SYNTHETIC_CORR)
    cov_ann = np.outer(vol_ann, vol_ann) * corr

    # Convert annual parameters to monthly frequency.
    mu_m = mu_ann / MONTHS_PER_YEAR
    cov_m = cov_ann / MONTHS_PER_YEAR

    # Month-end index from January 2015 through December 2025.
    dates = pd.date_range("2015-01-31", "2025-12-31", freq="ME")
    n_months = len(dates)

    draws = rng.multivariate_normal(mu_m, cov_m, size=n_months)
    rets = pd.DataFrame(draws, index=dates, columns=names)
    return rets


def load_returns_from_csv(csv_path, contains_prices=False):
    """Load a user-supplied CSV of monthly returns or prices.

    Expected schema: a date column (first column, parseable by pandas) and one
    numeric column per asset. If ``contains_prices`` is True the data are
    treated as adjusted prices and converted to simple monthly returns via
    percentage changes.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    contains_prices : bool
        Whether the CSV holds prices (True) or returns (False).

    Returns
    -------
    pandas DataFrame
        Clean monthly simple returns sorted ascending by date.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # The first column is interpreted as the date column.
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="any")
    if contains_prices:
        if (df <= 0).any().any():
            raise ValueError("Price data must be strictly positive.")
        df = df.pct_change().dropna()
    if df.shape[0] < 36:
        raise ValueError(
            "Need at least 36 months of history for estimation and walk-forward."
        )
    if df.shape[1] < 8 or df.shape[1] > 12:
        raise ValueError("The project targets between 8 and 12 asset columns.")
    return df


def download_monthly_returns_yfinance():
    """Download monthly adjusted prices via yfinance and convert to returns.

    Uses the ten tickers in ASSET_UNIVERSE over 2015 to 2025. Requires the
    optional yfinance package and internet access. Raises a clear error when
    either is unavailable so the user can fall back to synthetic or csv mode.
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is not installed. Install it with 'pip install yfinance' "
            "or use --mode synthetic or --mode csv."
        )
    tickers = [a[1] for a in ASSET_UNIVERSE]
    names = [a[0] for a in ASSET_UNIVERSE]
    raw = yf.download(
        tickers, start="2014-12-01", end="2026-01-01", auto_adjust=True,
        progress=False,
    )["Close"]
    if raw.empty:
        raise RuntimeError("yfinance returned no data; check connectivity.")
    raw = raw.dropna(how="any")
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change().dropna()
    # yfinance may reorder ticker columns; realign to the universe order
    # explicitly before renaming to display names so no asset is mislabeled.
    available = [t for t in tickers if t in rets.columns]
    rets = rets[available]
    ticker_to_name = dict(zip(tickers, names))
    rets.columns = [ticker_to_name[t] for t in available]
    return rets


def validate_returns(rets):
    """Run structural sanity checks on the monthly return matrix.

    Checks: no missing or infinite values, at least 8 and at most 12 assets,
    enough history for the estimation window, and no nonpositive sample
    volatility (which would be impossible for a traded asset).
    """
    arr = np.asarray(rets)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Return matrix contains NaN or infinite values.")
    if not (8 <= rets.shape[1] <= 12):
        raise ValueError("Universe must contain between 8 and 12 assets.")
    if rets.shape[0] < ESTIMATION_WINDOW + REBALANCE_STEP:
        raise ValueError("Not enough history for the walk-forward design.")
    sample_vol = arr.std(axis=0, ddof=1)
    if np.any(sample_vol <= 0):
        raise ValueError("An asset has nonpositive sample volatility.")
    return True


# =============================================================================
# Section 2: Covariance estimation and equilibrium (inverse optimization)
# =============================================================================

def shrink_covariance(sample_cov, returns=None):
    """Estimate a shrunk, well-conditioned covariance matrix.

    Primary path: Ledoit-Wolf shrinkage toward a scaled identity target, using
    scikit-learn when available. Fallback path: a closed-form shrinkage
    estimator with intensity chosen as 10 percent, shrinking the sample matrix
    toward its average-variance scaled identity.

    Shrinking covariance estimates is one of the two classical defenses
    against estimation error in mean-variance analysis (the other, used in
    this project, is anchoring expected returns to equilibrium).

    Parameters
    ----------
    sample_cov : (n, n) ndarray
        Sample covariance matrix of the returns.
    returns : (T, n) ndarray or None
        Raw returns, needed only by the scikit-learn Ledoit-Wolf path.

    Returns
    -------
    (n, n) ndarray
        Shrunk covariance matrix, guaranteed positive semidefinite by
        construction (convex combination of two PSD matrices).
    """
    n = sample_cov.shape[0]
    if returns is not None and SKLEARN_AVAILABLE:
        lw = LedoitWolf().fit(returns)
        shrunk = lw.covariance_
    else:
        # Fallback: fixed-intensity shrinkage toward the scaled identity.
        intensity = 0.10
        target = (np.trace(sample_cov) / n) * np.eye(n)
        shrunk = (1.0 - intensity) * sample_cov + intensity * target
    # Symmetrize to remove floating-point asymmetry.
    return 0.5 * (shrunk + shrunk.T)


def annualized_shrunk_covariance(window_returns):
    """Convenience wrapper: sample covariance, shrinkage, annualization.

    Parameters
    ----------
    window_returns : (T, n) ndarray or DataFrame
        Monthly returns over the estimation window.

    Returns
    -------
    (n, n) ndarray
        Annualized shrunk covariance matrix.
    """
    arr = np.asarray(window_returns)
    sample_cov = np.cov(arr, rowvar=False, ddof=1)
    shrunk = shrink_covariance(sample_cov, returns=arr)
    return shrunk * MONTHS_PER_YEAR


def equilibrium_returns(cov_ann, w_mkt, delta=RISK_AVERSION):
    """Recover implied equilibrium expected returns by inverse optimization.

    If the market portfolio w_mkt is mean-variance optimal for an average
    investor with risk aversion delta, the first-order condition of the
    quadratic utility program implies

        pi = delta * Sigma * w_mkt,

    where pi is the vector of equilibrium expected (excess) returns. Reverse
    engineering pi in this way replaces noisy sample means with a stable,
    economically motivated starting point.

    Parameters
    ----------
    cov_ann : (n, n) ndarray
        Annualized covariance matrix.
    w_mkt : (n,) ndarray
        Market-capitalization style weights (must sum to one).
    delta : float
        Risk aversion of the representative investor.

    Returns
    -------
    (n,) ndarray
        Implied equilibrium annual expected returns.
    """
    w_mkt = np.asarray(w_mkt, dtype=float)
    if not np.isclose(w_mkt.sum(), 1.0, atol=1e-8):
        raise ValueError("Market weights must sum to one.")
    if np.any(w_mkt < 0):
        raise ValueError("Market weights must be nonnegative.")
    return delta * cov_ann @ w_mkt


# =============================================================================
# Section 3: Investor views and the Black-Litterman posterior
# =============================================================================

def build_views(asset_names):
    """Construct a realistic set of economic views on the asset universe.

    Each view is one row of the (K, n) pick matrix P acting on the asset
    expected return vector, together with a scalar forecast in Q and a
    confidence level in (0, 1] used to scale the view uncertainty.

    The demo views below are deliberately generic and economically motivated:

        View 1 (relative): US large caps outperform developed ex-US equities
            by 2.0 percent per year (medium confidence).
        View 2 (absolute): US aggregate bonds return 3.0 percent per year
            (medium-high confidence).
        View 3 (relative): emerging market equities outperform US small caps
            by 3.0 percent per year (lower confidence).

    Parameters
    ----------
    asset_names : list of str
        Asset display names, used to locate view positions by name.

    Returns
    -------
    P : (K, n) ndarray
        Pick matrix; each row encodes one view.
    Q : (K,) ndarray
        View forecasts (annual returns or return spreads).
    confidence : (K,) ndarray
        Confidence per view in (0, 1]; one is maximal confidence.
    view_descriptions : list of str
        Human-readable description of each view for reporting.
    """
    idx = {name: i for i, name in enumerate(asset_names)}
    n = len(asset_names)

    P = np.zeros((3, n))
    # View 1: long US large cap, short developed ex-US (relative view, +2.0%).
    P[0, idx["US Large Cap Equity"]] = 1.0
    P[0, idx["Developed ex-US Equity"]] = -1.0
    # View 2: absolute view on US aggregate bonds (+3.0%).
    P[1, idx["US Aggregate Bonds"]] = 1.0
    # View 3: long emerging markets, short US small cap (relative view, +3.0%).
    P[2, idx["Emerging Markets Equity"]] = 1.0
    P[2, idx["US Small Cap Equity"]] = -1.0

    Q = np.array([0.020, 0.030, 0.030])
    confidence = np.array([0.50, 0.60, 0.35])
    descriptions = [
        "US Large Cap beats Developed ex-US Equity by 2.0% per year",
        "US Aggregate Bonds return 3.0% per year (absolute view)",
        "Emerging Markets Equity beats US Small Cap Equity by 3.0% per year",
    ]
    return P, Q, confidence, descriptions


def view_uncertainty(P, cov_ann, confidence, tau=TAU):
    """Build the view uncertainty matrix Omega.

    View uncertainty is set proportional to the variance of each view
    portfolio under the asset covariance matrix:

        Omega_k = (1 / c_k) * (P * Sigma * P^T)_{kk},

    where c_k is the analyst's confidence in view k. Higher confidence means
    lower Omega and therefore stronger influence of the view on the posterior.
    As confidence tends to zero, Omega tends to infinity and the view carries
    no information; as confidence tends to infinity, Omega tends to zero and
    the view binds exactly.

    Note on identification: tau is deliberately kept OUT of Omega here. If
    Omega were also scaled by tau (the original He and Litterman convention),
    tau would cancel out of the posterior mean and could not be identified.
    Keeping Omega proportional to P Sigma P^T through the confidence channel
    alone makes tau a genuine, separately interpretable control on the weight
    given to the equilibrium prior, which is also what makes the tau
    sensitivity study below informative.

    Parameters
    ----------
    P : (K, n) ndarray
        Pick matrix.
    cov_ann : (n, n) ndarray
        Annualized asset covariance matrix.
    confidence : (K,) ndarray
        Per-view confidence in (0, 1] (large values above one are also
        accepted for the infinite-confidence limiting-case test).
    tau : float
        Unused here; retained in the signature for interface compatibility.
        Prior uncertainty is applied inside black_litterman_posterior.

    Returns
    -------
    (K, K) ndarray
        Diagonal view uncertainty matrix.
    """
    confidence = np.asarray(confidence, dtype=float)
    if np.any(confidence <= 0):
        raise ValueError("Confidence must be strictly positive; use the "
                         "zero-confidence limiting helper for c -> 0.")
    view_var = np.diag(P @ cov_ann @ P.T)
    omega_diag = view_var / confidence
    return np.diag(omega_diag)


def black_litterman_posterior(pi, cov_ann, P, Q, omega, tau=TAU):
    """Compute the Black-Litterman posterior expected returns and covariance.

    The prior distribution of the unknown mean vector mu is centered at the
    equilibrium vector pi with covariance tau * Sigma. The views state that
    P * mu = Q + epsilon with epsilon ~ N(0, Omega). Conjugacy of the normal
    model gives the closed-form posterior

        mu_BL    = M * [ (tau Sigma)^{-1} pi + P^T Omega^{-1} Q ],
        Sigma_mu = M,
        M        = [ (tau Sigma)^{-1} + P^T Omega^{-1} P ]^{-1},

    and the posterior distribution of asset returns about the mean is
    Sigma_BL = Sigma + M (Sigma itself is treated as known).

    Numerical implementation: rather than inverting the n by n precision
    matrices above, we use the algebraically equivalent Kalman update form,

        A      = tau * Sigma,
        S      = P A P^T + Omega,
        mu_BL  = pi + A P^T S^{-1} (Q - P pi),
        M      = A - A P^T S^{-1} P A,

    which only requires solving a K by K system (K is the number of views,
    typically far smaller than n). This form remains accurate in the limiting
    cases Omega -> 0 (views bind exactly) and Omega -> infinity (posterior
    collapses onto the prior), where the precision form becomes ill
    conditioned.

    Parameters
    ----------
    pi : (n,) ndarray
        Equilibrium (prior mean) annual expected returns.
    cov_ann : (n, n) ndarray
        Annualized asset covariance matrix Sigma.
    P : (K, n) ndarray
        Pick matrix of the views.
    Q : (K,) ndarray
        View forecasts.
    omega : (K, K) ndarray
        View uncertainty matrix; may be exactly zero for infinite confidence.
    tau : float
        Prior uncertainty scale.

    Returns
    -------
    mu_bl : (n,) ndarray
        Posterior annual expected returns.
    cov_bl : (n, n) ndarray
        Posterior covariance of asset returns, Sigma + M.
    posterior_mean_cov : (n, n) ndarray
        Posterior covariance M of the mean estimate itself.
    """
    A = tau * cov_ann
    S = P @ A @ P.T + omega
    S = 0.5 * (S + S.T)  # symmetrize the small K by K system

    # Solve S X = (Q - P pi) for the Kalman-style update direction, and
    # S Y = P A for the covariance update, using linear solves rather than
    # explicit inversion for numerical stability.
    innovation = Q - P @ pi
    alpha = np.linalg.solve(S, innovation)        # S^{-1} (Q - P pi)
    mu_bl = pi + A @ P.T @ alpha

    Y = np.linalg.solve(S, P @ A)                 # S^{-1} P A
    post_cov_mean = A - A @ P.T @ Y

    cov_bl = cov_ann + post_cov_mean
    # Symmetrize outputs to remove floating-point asymmetry.
    cov_bl = 0.5 * (cov_bl + cov_bl.T)
    post_cov_mean = 0.5 * (post_cov_mean + post_cov_mean.T)
    return mu_bl, cov_bl, post_cov_mean


# =============================================================================
# Section 4: Constrained mean-variance optimization
# =============================================================================

def mean_variance_optimize(mu, cov_ann, delta=RISK_AVERSION, cap=ASSET_CAP):
    """Solve the constrained mean-variance allocation problem.

    The program is

        maximize    w^T mu - (delta / 2) * w^T Sigma w
        subject to  sum(w) = 1,
                    0 <= w_i <= cap   for all i.

    The primary solver is cvxpy (a disciplined convex programming interface).
    When cvxpy is unavailable, a scipy SLSQP fallback with an analytic
    gradient is used. Both paths are checked against the same constraint
    tolerances before the weights are returned.

    Parameters
    ----------
    mu : (n,) ndarray
        Annual expected returns.
    cov_ann : (n, n) ndarray
        Annualized covariance matrix.
    delta : float
        Risk aversion parameter of the utility function.
    cap : float
        Per-asset upper weight bound.

    Returns
    -------
    (n,) ndarray
        Optimal long-only weights summing to one.
    """
    mu = np.asarray(mu, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    n = len(mu)
    if n * cap < 1.0 - 1e-12:
        raise ValueError("Per-asset cap is infeasible: n * cap must be >= 1.")

    if CVXPY_AVAILABLE:
        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - (delta / 2.0) * cp.quad_form(w, cov_ann))
        constraints = [cp.sum(w) == 1.0, w >= 0.0, w <= cap]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver="OSQP")
        except Exception:
            problem.solve()  # fall back to the default solver
        if w.value is None:
            raise RuntimeError("cvxpy failed to find an optimal solution.")
        weights = np.asarray(w.value).flatten()
    else:
        # scipy SLSQP fallback: minimize the negative utility.
        def neg_utility(w):
            return -(w @ mu) + 0.5 * delta * w @ cov_ann @ w

        def neg_utility_grad(w):
            return -mu + delta * cov_ann @ w

        w0 = np.full(n, 1.0 / n)  # feasible equal-weight start
        bounds = [(0.0, cap)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                 "jac": lambda w: np.ones_like(w)}]
        result = minimize(
            neg_utility, w0, jac=neg_utility_grad, bounds=bounds,
            constraints=cons, method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP optimization failed: {result.message}")
        weights = result.x

    # Clean tiny numerical noise and renormalize defensively.
    weights[np.abs(weights) < 1e-10] = 0.0
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    # Hard constraint check: weights sum to one and respect bounds.
    assert abs(weights.sum() - 1.0) < TOL_CONSTRAINT, "weights must sum to one"
    assert weights.min() >= -TOL_CONSTRAINT, "long-only constraint violated"
    assert weights.max() <= cap + TOL_CONSTRAINT, "per-asset cap violated"
    return weights


def mean_variance_unconstrained(mu, cov_ann, delta=RISK_AVERSION):
    """Solve the fully-invested mean-variance problem with no sign or cap
    constraints (shorting and leverage above the cap are both permitted).

    This is the textbook Lagrangian solution to
        maximize    w^T mu - (delta / 2) * w^T Sigma w
        subject to  sum(w) = 1
    obtained in closed form as
        w* = (1/delta) * Sigma^{-1} * (mu - lambda * 1),
        lambda chosen so that 1^T w* = 1.

    It serves as the "no constraints" reference point for the constraint-cost
    analysis below: the utility this portfolio achieves is the best any
    long-only, capped, or tracking-error-limited portfolio could ever match,
    so the gap is a clean measure of how much the constraint set costs.

    Parameters
    ----------
    mu : (n,) ndarray
        Annual expected returns.
    cov_ann : (n, n) ndarray
        Annualized covariance matrix.
    delta : float
        Risk aversion parameter.

    Returns
    -------
    (n,) ndarray
        Unconstrained (possibly negative or >cap) optimal weights.
    """
    n = len(mu)
    ones = np.ones(n)
    cov_inv_mu = np.linalg.solve(cov_ann, mu)
    cov_inv_1 = np.linalg.solve(cov_ann, ones)
    lam = (ones @ cov_inv_mu - delta) / (ones @ cov_inv_1)
    w = (cov_inv_mu - lam * cov_inv_1) / delta
    return w


def mean_variance_optimize_te(mu, cov_ann, w_bench, delta=RISK_AVERSION,
                              cap=ASSET_CAP, te_budget=TE_BUDGET):
    """Solve the constrained mean-variance problem with an added tracking-
    error budget relative to a reference benchmark portfolio.

    The program is the same long-only, capped, fully-invested problem as
    ``mean_variance_optimize`` with one additional constraint:

        sqrt[(w - w_bench)^T Sigma (w - w_bench)] <= te_budget

    Tracking-error budgets are the standard institutional mechanism for
    letting an active manager deviate from a policy benchmark only up to a
    pre-agreed risk tolerance. This is the "tracking-error-constrained
    variant" of the optimizer referenced in the project specification,
    complementing the pure per-asset-cap formulation used elsewhere.

    Parameters
    ----------
    mu : (n,) ndarray
        Annual expected returns (typically the BL posterior).
    cov_ann : (n, n) ndarray
        Annualized covariance matrix.
    w_bench : (n,) ndarray
        Reference benchmark weights (e.g. the 60/40 portfolio).
    delta : float
        Risk aversion parameter.
    cap : float
        Per-asset upper weight bound.
    te_budget : float
        Maximum permitted annualized tracking-error volatility.

    Returns
    -------
    (n,) ndarray
        Optimal weights subject to the tracking-error budget.
    """
    mu = np.asarray(mu, dtype=float)
    cov_ann = np.asarray(cov_ann, dtype=float)
    w_bench = np.asarray(w_bench, dtype=float)
    n = len(mu)

    if CVXPY_AVAILABLE:
        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - (delta / 2.0) * cp.quad_form(w, cov_ann))
        diff = w - w_bench
        constraints = [
            cp.sum(w) == 1.0, w >= 0.0, w <= cap,
            cp.quad_form(diff, cov_ann) <= te_budget ** 2,
        ]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver="OSQP")
        except Exception:
            problem.solve()
        if w.value is None:
            raise RuntimeError("cvxpy failed to find a TE-constrained solution.")
        weights = np.asarray(w.value).flatten()
    else:
        # scipy SLSQP fallback: same objective, with an added nonlinear
        # inequality constraint on the tracking-error variance.
        def neg_utility(w):
            return -(w @ mu) + 0.5 * delta * w @ cov_ann @ w

        def neg_utility_grad(w):
            return -mu + delta * cov_ann @ w

        def te_slack(w):
            diff = w - w_bench
            te_var = diff @ cov_ann @ diff
            return te_budget ** 2 - te_var  # must be >= 0

        w0 = np.clip(w_bench.copy(), 0.0, cap)
        if not np.isclose(w0.sum(), 1.0):
            w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, cap)] * n
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": te_slack},
        ]
        result = minimize(
            neg_utility, w0, jac=neg_utility_grad, bounds=bounds,
            constraints=cons, method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"SLSQP TE-constrained optimization failed: "
                               f"{result.message}")
        weights = result.x

    weights[np.abs(weights) < 1e-10] = 0.0
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    assert abs(weights.sum() - 1.0) < TOL_CONSTRAINT, "weights must sum to one"
    assert weights.min() >= -TOL_CONSTRAINT, "long-only constraint violated"
    assert weights.max() <= cap + TOL_CONSTRAINT, "per-asset cap violated"
    return weights


def constraint_cost_table(mu, cov_ann, w_bench, delta=RISK_AVERSION,
                          cap=ASSET_CAP, te_budget=TE_BUDGET):
    """Quantify the utility cost of the project's constraint set.

    Three variants of the same mean-variance problem (identical mu and
    Sigma, the BL posterior) are solved with progressively tighter
    constraint sets, and their quadratic utility, expected return, realized
    volatility, and tracking error against the benchmark are compared to the
    fully unconstrained (shorting and leverage allowed) optimum. The utility
    gap to the unconstrained case is the "price" paid for that constraint,
    expressed in the same annualized-return units as mu.

        1. Unconstrained : sum(w) = 1 only (shorting and leverage allowed).
        2. Long-only + cap : the project's default constraint set.
        3. Long-only + cap + tracking-error budget vs the benchmark.

    Parameters
    ----------
    mu : (n,) ndarray
        Annual expected returns (the BL posterior in the base case).
    cov_ann : (n, n) ndarray
        Annualized covariance matrix.
    w_bench : (n,) ndarray
        Reference benchmark weights for the tracking-error variant
        (the 60/40 benchmark).
    delta, cap, te_budget : float
        Model and constraint parameters.

    Returns
    -------
    pandas DataFrame
        One row per variant with utility, expected return, volatility,
        tracking error, max weight, and utility cost versus unconstrained.
    """
    def summarize(label, w):
        exp_ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov_ann @ w))
        utility = exp_ret - 0.5 * delta * float(w @ cov_ann @ w)
        diff = w - w_bench
        te = float(np.sqrt(max(diff @ cov_ann @ diff, 0.0)))
        return {
            "Variant": label,
            "Expected Return": exp_ret,
            "Volatility": vol,
            "Utility": utility,
            "Tracking Error vs 60/40": te,
            "Max Weight": float(w.max()),
            "Min Weight": float(w.min()),
        }

    w_unc = mean_variance_unconstrained(mu, cov_ann, delta=delta)
    w_cap = mean_variance_optimize(mu, cov_ann, delta=delta, cap=cap)
    w_te = mean_variance_optimize_te(mu, cov_ann, w_bench, delta=delta,
                                     cap=cap, te_budget=te_budget)

    rows = [
        summarize("Unconstrained (shorting and leverage allowed)", w_unc),
        summarize("Long-only + 25% per-asset cap (project default)", w_cap),
        summarize(f"Long-only + cap + {te_budget:.0%} TE budget vs 60/40",
                  w_te),
    ]
    df = pd.DataFrame(rows)
    u0 = df.loc[0, "Utility"]
    df["Utility Cost vs Unconstrained"] = u0 - df["Utility"]
    return df


# =============================================================================
# Section 5: Naive mean-variance failure analysis
# =============================================================================

def bootstrap_instability(rets, w_mkt, n_draws=6, seed=RANDOM_SEED,
                          delta=RISK_AVERSION, tau=TAU, cap=ASSET_CAP):
    """Measure estimation-error sensitivity by bootstrap resampling.

    The classic failure mode of naive mean-variance optimization is error
    maximization: small changes in the estimated inputs produce large swings
    in the optimal portfolio, because the optimizer treats noisy sample means
    as if they were known truths. With long-only constraints and caps this
    instability appears as corner solutions that flip between assets when the
    estimation sample changes, which a static perturbation of the mean vector
    does not reveal (corner solutions are locally sticky). We therefore follow
    the resampling tradition of Michaud (1998): draw bootstrap resamples of
    the monthly return panel, re-estimate each model, and measure the average
    L1 distance between resampled and base-case optimal weights.

    Parameters
    ----------
    rets : pandas DataFrame
        Monthly simple returns used for estimation.
    w_mkt : (n,) ndarray
        Market weights used by the Black-Litterman equilibrium prior.
    n_draws : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mean L1 weight distances for the naive and Black-Litterman models.
    """
    names = list(rets.columns)
    arr = np.asarray(rets)
    T = arr.shape[0]
    rng = np.random.default_rng(seed)
    P, Q, confidence, _ = build_views(names)

    # Base-case weights estimated on the full panel.
    cov_full = annualized_shrunk_covariance(arr)
    pi = equilibrium_returns(cov_full, w_mkt, delta=delta)
    omega = view_uncertainty(P, cov_full, confidence)
    mu_bl, _, _ = black_litterman_posterior(pi, cov_full, P, Q, omega, tau=tau)
    w_bl_base = mean_variance_optimize(mu_bl, cov_full, delta=delta, cap=cap)

    mu_sample = arr.mean(axis=0) * MONTHS_PER_YEAR
    cov_sample = np.cov(arr, rowvar=False, ddof=1) * MONTHS_PER_YEAR
    w_naive_base = mean_variance_optimize(mu_sample, cov_sample, delta=delta,
                                          cap=cap)

    dist_naive, dist_bl = [], []
    for _ in range(n_draws):
        idx = rng.integers(0, T, T)  # bootstrap resample of months
        sample = arr[idx]

        # Naive leg: raw sample moments of the resample.
        mu_n = sample.mean(axis=0) * MONTHS_PER_YEAR
        cov_n = np.cov(sample, rowvar=False, ddof=1) * MONTHS_PER_YEAR
        w_n = mean_variance_optimize(mu_n, cov_n, delta=delta, cap=cap)
        dist_naive.append(np.abs(w_n - w_naive_base).sum())

        # Black-Litterman leg: shrunk covariance, equilibrium prior, views.
        cov_b = annualized_shrunk_covariance(sample)
        pi_b = equilibrium_returns(cov_b, w_mkt, delta=delta)
        omega_b = view_uncertainty(P, cov_b, confidence)
        mu_b, _, _ = black_litterman_posterior(pi_b, cov_b, P, Q, omega_b,
                                               tau=tau)
        w_b = mean_variance_optimize(mu_b, cov_b, delta=delta, cap=cap)
        dist_bl.append(np.abs(w_b - w_bl_base).sum())

    return {
        "naive_mean_L1": float(np.mean(dist_naive)),
        "bl_mean_L1": float(np.mean(dist_bl)),
        "naive_draws": dist_naive,
        "bl_draws": dist_bl,
    }


def effective_number_of_bets(weights):
    """Effective number of bets, defined as the inverse Herfindahl index.

    A portfolio fully concentrated in one asset scores one; an equal-weight
    portfolio of n assets scores n. This is a simple concentration diagnostic.
    """
    w = np.asarray(weights, dtype=float)
    return 1.0 / np.sum(w ** 2)


# =============================================================================
# Section 6: Unit tests of correctness and limiting cases
# =============================================================================

def run_unit_tests(verbose=True):
    """Run the correctness and limiting-case test suite.

    Tests implemented (each returns a (name, passed, detail) tuple):

        1. Zero-confidence views recover equilibrium exactly: with Omega
           scaled toward infinity the posterior mean equals the prior pi.
        2. Infinite-confidence views satisfy P * mu = Q exactly: with Omega
           set to zero the posterior mean meets every view without error.
        3. tau -> 0 recovers the prior: with an arbitrarily confident prior
           the posterior mean equals pi.
        4. Equilibrium risk premia are positive and plausible.
        5. Optimizer constraints hold (sum to one, long-only, caps).
        6. Covariance matrices are positive semidefinite.

    Parameters
    ----------
    verbose : bool
        Print a formatted test report when True.

    Returns
    -------
    bool
        True if and only if every test passed.
    """
    results = []

    # Build a small, fully controlled test environment on the synthetic
    # calibration so the tests are deterministic and fast.
    names = [a[0] for a in ASSET_UNIVERSE]
    vol_ann = np.array([a[3] for a in ASSET_UNIVERSE])
    corr = nearest_psd_correlation(SYNTHETIC_CORR)
    cov_ann = np.outer(vol_ann, vol_ann) * corr
    w_mkt = SYNTHETIC_MARKET_WEIGHTS
    pi = equilibrium_returns(cov_ann, w_mkt)
    P, Q, confidence, _ = build_views(names)
    omega = view_uncertainty(P, cov_ann, confidence)

    # --- Test 1: zero-confidence views recover equilibrium exactly. ---------
    # Scale Omega by a huge factor, which is the operational meaning of zero
    # confidence: the views carry no information.
    mu_low, _, _ = black_litterman_posterior(pi, cov_ann, P, Q, omega * 1e12)
    err = np.max(np.abs(mu_low - pi))
    results.append((
        "Zero-confidence views recover equilibrium exactly",
        err < TOL_LIMITING_CASE,
        f"max |mu_BL - pi| = {err:.3e}",
    ))

    # --- Test 2: infinite-confidence views satisfy P mu = Q exactly. --------
    # With Omega = 0 the posterior mean is the prior projected onto the view
    # constraints, so P mu_BL = Q holds to machine precision. The Kalman
    # update form handles Omega = 0 exactly because S = P (tau Sigma) P^T
    # remains invertible when the view rows are linearly independent.
    mu_high, _, _ = black_litterman_posterior(pi, cov_ann, P, Q,
                                              np.zeros((len(Q), len(Q))))
    err = np.max(np.abs(P @ mu_high - Q))
    results.append((
        "Infinite-confidence views satisfy P mu = Q",
        err < 1e-6,
        f"max |P mu_BL - Q| = {err:.3e}",
    ))

    # --- Test 3: tau -> 0 recovers the prior. -------------------------------
    mu_tau0, _, _ = black_litterman_posterior(
        pi, cov_ann, P, Q, omega, tau=1e-12
    )
    err = np.max(np.abs(mu_tau0 - pi))
    results.append((
        "tau approaching zero recovers equilibrium prior",
        err < 1e-6,
        f"max |mu_BL(tau->0) - pi| = {err:.3e}",
    ))

    # --- Test 4: equilibrium risk premia are positive and plausible. --------
    plausible = np.all(pi > 0.0) and np.all(pi < 0.20)
    results.append((
        "Equilibrium risk premia positive and plausible (0% to 20%)",
        plausible,
        f"pi range = [{pi.min():.3%}, {pi.max():.3%}]",
    ))

    # --- Test 5: optimizer constraints hold on both solver paths. -----------
    mu_bl, cov_bl, _ = black_litterman_posterior(pi, cov_ann, P, Q, omega)
    w_opt = mean_variance_optimize(mu_bl, cov_ann)
    constraints_ok = (
        abs(w_opt.sum() - 1.0) < TOL_CONSTRAINT
        and w_opt.min() >= -TOL_CONSTRAINT
        and w_opt.max() <= ASSET_CAP + TOL_CONSTRAINT
    )
    results.append((
        "Optimizer constraints: sum to one, long-only, per-asset cap",
        constraints_ok,
        f"sum = {w_opt.sum():.6f}, min = {w_opt.min():.4f}, "
        f"max = {w_opt.max():.4f}, cap = {ASSET_CAP}",
    ))

    # --- Test 6: covariance matrices are positive semidefinite. -------------
    eig_sample = np.linalg.eigvalsh(cov_ann).min()
    eig_post = np.linalg.eigvalsh(cov_bl).min()
    psd_ok = eig_sample > TOL_PSD and eig_post > TOL_PSD
    results.append((
        "Covariance matrices positive semidefinite",
        psd_ok,
        f"min eigenvalue Sigma = {eig_sample:.3e}, "
        f"Sigma_BL = {eig_post:.3e}",
    ))

    if verbose:
        print("\n" + "=" * 72)
        print("UNIT TEST REPORT: Black-Litterman correctness and limiting cases")
        print("=" * 72)
        for name, passed, detail in results:
            flag = "PASS" if passed else "FAIL"
            print(f"[{flag}] {name}\n       {detail}")
        print("=" * 72)

    return all(r[1] for r in results), results


# =============================================================================
# Section 7: Performance metrics and walk-forward backtest
# =============================================================================

def portfolio_metrics(monthly_returns, label):
    """Compute standard out-of-sample performance statistics.

    Parameters
    ----------
    monthly_returns : pandas Series
        Out-of-sample monthly portfolio returns.
    label : str
        Strategy name for the output row.

    Returns
    -------
    dict
        CAGR, annualized volatility, Sharpe ratio (zero risk-free rate),
        and maximum drawdown.
    """
    r = np.asarray(monthly_returns, dtype=float)
    n_months = len(r)
    wealth = np.cumprod(1.0 + r)
    total_return = wealth[-1] - 1.0
    years = n_months / MONTHS_PER_YEAR
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    # Volatility is the square root of a variance and can never be negative.
    vol_ann = np.std(r, ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    sharpe = (np.mean(r) * MONTHS_PER_YEAR) / vol_ann if vol_ann > 0 else np.nan
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    max_dd = drawdown.min()
    return {
        "Strategy": label,
        "CAGR": cagr,
        "Ann. Volatility": vol_ann,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": max_dd,
    }


def run_walk_forward(rets, w_mkt, delta=RISK_AVERSION, tau=TAU, cap=ASSET_CAP,
                     window=ESTIMATION_WINDOW, step=REBALANCE_STEP,
                     start_date=WALK_FORWARD_START, verbose=True):
    """Quarterly walk-forward comparison of four allocation strategies.

    Strategies compared out of sample, from 2020 onward:

        1. Black-Litterman : equilibrium prior + views, shrunk covariance.
        2. Naive MV        : raw sample means and raw sample covariance.
        3. 60/40 benchmark : 60 percent US large cap, 40 percent US aggregate
           bonds, rebalanced quarterly.
        4. Equal weight    : 1/n across all assets, rebalanced quarterly.

    At each quarterly rebalance date the estimation window is the trailing
    ``window`` months; portfolios are held fixed between rebalances. Turnover
    is measured as the L1 weight change at each rebalance, accounting for
    weight drift over the holding period.

    Parameters
    ----------
    rets : pandas DataFrame
        Monthly simple returns for the full sample.
    w_mkt : (n,) ndarray
        Market-cap style weights used for the equilibrium prior.
    delta, tau, cap, window, step : float / int
        Model and design parameters.
    start_date : str
        First out-of-sample month.

    Returns
    -------
    metrics_df : pandas DataFrame
        Performance statistics per strategy.
    wealth_df : pandas DataFrame
        Out-of-sample cumulative wealth paths per strategy.
    turnover_df : pandas DataFrame
        Per-rebalance turnover for the optimized strategies.
    weights_log : dict
        Rebalance-date-indexed weight history per optimized strategy.
    """
    names = list(rets.columns)
    n = len(names)
    dates = rets.index
    start_pos = dates.searchsorted(pd.Timestamp(start_date))
    if start_pos < window:
        start_pos = window
    # Rebalance at every ``step`` months starting from start_pos, always
    # leaving at least one holding period ahead.
    rebalance_positions = list(range(start_pos, len(dates) - step + 1, step))
    if len(rebalance_positions) < 2:
        raise ValueError("Not enough out-of-sample quarters for walk-forward.")

    P, Q, confidence, _ = build_views(names)

    # Static benchmark weight vectors.
    w_equal = np.full(n, 1.0 / n)
    w_6040 = np.zeros(n)
    w_6040[names.index("US Large Cap Equity")] = 0.60
    w_6040[names.index("US Aggregate Bonds")] = 0.40

    strategies = ["Black-Litterman", "Naive MV", "60/40 Benchmark",
                  "Equal Weight"]
    # Per-strategy containers.
    period_returns = {s: [] for s in strategies}
    period_dates = []
    turnover_records = []
    weights_log = {"Black-Litterman": {}, "Naive MV": {}}

    # Current holdings, initialized at the first rebalance date.
    current = {s: None for s in strategies}

    for k, pos in enumerate(rebalance_positions):
        est = rets.iloc[pos - window: pos]
        est_arr = np.asarray(est)

        # --- Black-Litterman leg ------------------------------------------
        cov_shrunk = annualized_shrunk_covariance(est_arr)
        pi = equilibrium_returns(cov_shrunk, w_mkt, delta=delta)
        omega = view_uncertainty(P, cov_shrunk, confidence, tau=tau)
        mu_bl, _, _ = black_litterman_posterior(pi, cov_shrunk, P, Q, omega,
                                                tau=tau)
        w_bl = mean_variance_optimize(mu_bl, cov_shrunk, delta=delta, cap=cap)

        # --- Naive mean-variance leg --------------------------------------
        mu_sample = est_arr.mean(axis=0) * MONTHS_PER_YEAR
        cov_sample = np.cov(est_arr, rowvar=False, ddof=1) * MONTHS_PER_YEAR
        w_naive = mean_variance_optimize(mu_sample, cov_sample, delta=delta,
                                         cap=cap)

        # --- Record target weights and turnover ----------------------------
        new_targets = {"Black-Litterman": w_bl, "Naive MV": w_naive,
                       "60/40 Benchmark": w_6040, "Equal Weight": w_equal}
        for s in strategies:
            if current[s] is not None:
                turnover = np.abs(new_targets[s] - current[s]).sum()
            else:
                # Initial establishment of the portfolio counts as turnover.
                turnover = np.abs(new_targets[s]).sum()
            turnover_records.append({"Rebalance Date": dates[pos],
                                     "Strategy": s, "Turnover": turnover})
            current[s] = new_targets[s].copy()
        weights_log["Black-Litterman"][dates[pos]] = w_bl
        weights_log["Naive MV"][dates[pos]] = w_naive

        # --- Hold until the next rebalance (or the end of the sample) -----
        end_pos = (rebalance_positions[k + 1]
                   if k + 1 < len(rebalance_positions) else len(dates))
        holding = rets.iloc[pos:end_pos]
        for _, row in holding.iterrows():
            r_vec = row.values.astype(float)
            period_dates.append(row.name)
            for s in strategies:
                period_returns[s].append(float(current[s] @ r_vec))
                # Let weights drift with realized returns.
                current[s] = current[s] * (1.0 + r_vec)
                current[s] = current[s] / current[s].sum()

    # --- Assemble outputs ---------------------------------------------------
    idx = pd.DatetimeIndex(period_dates)
    metrics = []
    wealth = pd.DataFrame(index=idx)
    for s in strategies:
        series = pd.Series(period_returns[s], index=idx)
        metrics.append(portfolio_metrics(series, s))
        wealth[s] = np.cumprod(1.0 + series)
    metrics_df = pd.DataFrame(metrics)
    turnover_df = pd.DataFrame(turnover_records)

    # Average turnover per rebalance for each optimized strategy, excluding
    # the initial establishment trade.
    avg_turnover = (
        turnover_df[turnover_df["Rebalance Date"] != dates[start_pos]]
        .groupby("Strategy")["Turnover"].mean()
    )
    metrics_df["Avg Quarterly Turnover"] = metrics_df["Strategy"].map(
        avg_turnover
    )

    if verbose:
        print("\n" + "-" * 72)
        print("WALK-FORWARD OUT-OF-SAMPLE RESULTS (quarterly rebalancing)")
        print(f"Out-of-sample span: {idx.min().date()} to {idx.max().date()} "
              f"({len(idx)} months, {len(rebalance_positions)} rebalances)")
        print("-" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
            print(metrics_df.to_string(index=False))
    return metrics_df, wealth, turnover_df, weights_log


# =============================================================================
# Section 8: Sensitivity studies (tau and view confidence)
# =============================================================================

def tau_sensitivity(pi, cov_ann, P, Q, confidence, w_mkt,
                    tau_grid=(0.01, 0.025, 0.05, 0.10, 0.25, 0.50),
                    delta=RISK_AVERSION, cap=ASSET_CAP):
    """Study how the BL allocation responds to the prior scale tau.

    For each tau the posterior mean and optimal weights are recomputed. Two
    diagnostics are reported: the L1 distance of the posterior mean from the
    equilibrium prior, and the L1 distance of the BL weights from the
    equilibrium portfolio weights. Both should decline toward zero as tau
    shrinks, which is the correct theoretical limiting behavior.
    """
    rows = []
    w_eq = w_mkt / w_mkt.sum()
    for tau in tau_grid:
        omega = view_uncertainty(P, cov_ann, confidence, tau=tau)
        mu_bl, _, _ = black_litterman_posterior(pi, cov_ann, P, Q, omega,
                                                tau=tau)
        w_bl = mean_variance_optimize(mu_bl, cov_ann, delta=delta, cap=cap)
        rows.append({
            "tau": tau,
            "L1 |mu_BL - pi|": np.abs(mu_bl - pi).sum(),
            "L1 |w_BL - w_mkt|": np.abs(w_bl - w_eq).sum(),
            "Max asset weight": w_bl.max(),
        })
    return pd.DataFrame(rows)


def confidence_sensitivity(pi, cov_ann, P, Q, base_confidence, w_mkt,
                           scale_grid=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
                           tau=TAU, delta=RISK_AVERSION, cap=ASSET_CAP):
    """Study how the BL allocation responds to view confidence.

    The base confidence vector is multiplied by a scale factor. Low scales
    approximate the zero-confidence limit (posterior near equilibrium); high
    scales move the posterior toward the views. The same two L1 diagnostics as
    in the tau study are reported.
    """
    rows = []
    w_eq = w_mkt / w_mkt.sum()
    for scale in scale_grid:
        conf = np.clip(base_confidence * scale, 1e-8, None)
        omega = view_uncertainty(P, cov_ann, conf, tau=tau)
        mu_bl, _, _ = black_litterman_posterior(pi, cov_ann, P, Q, omega,
                                                tau=tau)
        w_bl = mean_variance_optimize(mu_bl, cov_ann, delta=delta, cap=cap)
        rows.append({
            "Confidence scale": scale,
            "L1 |mu_BL - pi|": np.abs(mu_bl - pi).sum(),
            "L1 |w_BL - w_mkt|": np.abs(w_bl - w_eq).sum(),
            "Max asset weight": w_bl.max(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Section 9: Plotting and output writers
# =============================================================================

def plot_wealth_paths(wealth_df, outdir, mode_label):
    """Plot out-of-sample cumulative wealth for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in wealth_df.columns:
        ax.plot(wealth_df.index, wealth_df[col], label=col, linewidth=1.8)
    ax.set_title(f"Walk-forward cumulative wealth ({mode_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 unit of wealth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "walkforward_cumulative_wealth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weight_comparison(names, w_mkt, w_bl, w_naive, outdir, mode_label):
    """Grouped bar chart comparing equilibrium, BL, and naive MV weights."""
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, w_mkt, width, label="Equilibrium (market)")
    ax.bar(x, w_bl, width, label="Black-Litterman")
    ax.bar(x + width, w_naive, width, label="Naive mean-variance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Portfolio weight")
    ax.set_title(f"Portfolio weights: equilibrium vs BL vs naive MV ({mode_label})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "weights_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_expected_returns(names, pi, mu_bl, sample_mu, outdir, mode_label):
    """Bar chart comparing sample means, equilibrium, and BL posterior means."""
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, sample_mu, width, label="Sample mean (noisy)")
    ax.bar(x, pi, width, label="Equilibrium pi")
    ax.bar(x + width, mu_bl, width, label="BL posterior")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Annual expected return")
    ax.set_title(f"Expected return estimates ({mode_label})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "expected_returns_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sensitivity(df, xcol, ycols, title, outdir, filename):
    """Generic line plot used by both sensitivity studies."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col in ycols:
        ax.plot(df[xcol], df[col], marker="o", label=col)
    ax.set_xscale("log")
    ax.set_xlabel(xcol)
    ax.set_ylabel("L1 distance")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_covariance_heatmap(cov_ann, names, outdir, mode_label):
    """Heatmap of the shrunk annualized correlation matrix."""
    vol = np.sqrt(np.maximum(np.diag(cov_ann), 0.0))
    corr = cov_ann / np.outer(vol, vol)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(f"Shrunk correlation matrix ({mode_label})")
    fig.tight_layout()
    path = os.path.join(outdir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bl_weight_evolution(weights_log, names, outdir, mode_label):
    """Stacked area chart of BL portfolio weights across walk-forward dates."""
    hist = weights_log.get("Black-Litterman", {})
    if not hist:
        return None
    w_df = pd.DataFrame(hist).T.sort_index()
    w_df.columns = names
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(w_df.index, w_df.T.values, labels=names, alpha=0.9)
    ax.set_title(f"Black-Litterman weights through time ({mode_label})")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Portfolio weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(outdir, "bl_weights_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# =============================================================================
# Section 10: Main driver
# =============================================================================

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Black-Litterman portfolio allocation project."
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "csv", "yfinance"],
        default="synthetic",
        help="Data source. 'synthetic' is a clearly labeled demonstration "
             "dataset; 'csv' reads a user file; 'yfinance' downloads data.",
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a CSV of monthly returns (or prices).")
    parser.add_argument("--prices", action="store_true",
                        help="Treat the CSV input as adjusted prices.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for figures and CSV outputs.")
    parser.add_argument("--selftest", action="store_true",
                        help="Run only the unit test suite and exit.")
    return parser.parse_args(argv)


def main(argv=None):
    """End-to-end driver: data, model, tests, backtest, sensitivity, outputs."""
    args = parse_args(argv)

    # The selftest entry point runs only the correctness suite.
    if args.selftest:
        ok, _ = run_unit_tests(verbose=True)
        print(f"\nSolver backend: "
              f"{'cvxpy' if CVXPY_AVAILABLE else 'scipy SLSQP fallback'}")
        return 0 if ok else 1

    # ------------------------------------------------------------------
    # Step 1: load or generate the monthly return panel.
    # ------------------------------------------------------------------
    if args.mode == "synthetic":
        rets = generate_synthetic_monthly_returns()
        mode_label = "SYNTHETIC DEMO DATA, not real market evidence"
        w_mkt = SYNTHETIC_MARKET_WEIGHTS
    elif args.mode == "csv":
        if args.csv is None:
            raise ValueError("--csv PATH is required in csv mode.")
        rets = load_returns_from_csv(args.csv, contains_prices=args.prices)
        mode_label = "user-supplied CSV data"
        # With real data and no external cap data, equal market weights are a
        # neutral stand-in; users should ideally supply true cap weights.
        w_mkt = np.full(rets.shape[1], 1.0 / rets.shape[1])
        print("NOTE: no market-cap weights supplied in csv mode; using equal "
              "weights for the equilibrium prior. Supply your own w_mkt for "
              "production use.")
    else:  # yfinance
        rets = download_monthly_returns_yfinance()
        mode_label = "yfinance adjusted-price data"
        w_mkt = SYNTHETIC_MARKET_WEIGHTS[: rets.shape[1]]
        w_mkt = w_mkt / w_mkt.sum()

    validate_returns(rets)
    names = list(rets.columns)
    n = len(names)

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print("BLACK-LITTERMAN PORTFOLIO ALLOCATION")
    print(f"Data mode: {args.mode} ({mode_label})")
    print(f"Universe: {n} assets, {len(rets)} monthly observations "
          f"({rets.index.min().date()} to {rets.index.max().date()})")
    print(f"Solver backend: "
          f"{'cvxpy' if CVXPY_AVAILABLE else 'scipy SLSQP fallback'}")
    print(f"delta = {RISK_AVERSION}, tau = {TAU}, per-asset cap = {ASSET_CAP}")
    print("=" * 72)

    # Persist the input data (clearly labeled in synthetic mode).
    data_name = ("SYNTHETIC_monthly_returns.csv" if args.mode == "synthetic"
                 else "input_monthly_returns.csv")
    rets.to_csv(os.path.join(outdir, data_name))

    # ------------------------------------------------------------------
    # Step 2: unit tests of limiting cases and correctness checks.
    # ------------------------------------------------------------------
    tests_ok, _ = run_unit_tests(verbose=True)
    if not tests_ok:
        print("WARNING: one or more unit tests failed; inspect the report.")

    # ------------------------------------------------------------------
    # Step 3: full-sample equilibrium, views, and BL posterior.
    # ------------------------------------------------------------------
    cov_full = annualized_shrunk_covariance(rets)
    pi = equilibrium_returns(cov_full, w_mkt)
    P, Q, confidence, view_descriptions = build_views(names)
    omega = view_uncertainty(P, cov_full, confidence)
    mu_bl, cov_bl, _ = black_litterman_posterior(pi, cov_full, P, Q, omega)

    sample_mu = np.asarray(rets).mean(axis=0) * MONTHS_PER_YEAR

    print("\nEquilibrium vs posterior expected returns (annualized):")
    er_table = pd.DataFrame({
        "Asset": names,
        "Sample mean": sample_mu,
        "Equilibrium pi": pi,
        "BL posterior": mu_bl,
    })
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(er_table.to_string(index=False))
    er_table.to_csv(os.path.join(outdir, "expected_returns_table.csv"),
                    index=False)

    print("\nInvestor views:")
    for desc, q, c in zip(view_descriptions, Q, confidence):
        print(f"  - {desc} [Q = {q:.3%}, confidence = {c:.0%}]")

    # ------------------------------------------------------------------
    # Step 4: optimize the BL, equilibrium, and naive MV portfolios.
    # ------------------------------------------------------------------
    w_bl = mean_variance_optimize(mu_bl, cov_full)
    w_eq = mean_variance_optimize(pi, cov_full)
    cov_sample_full = np.cov(np.asarray(rets), rowvar=False,
                             ddof=1) * MONTHS_PER_YEAR
    w_naive = mean_variance_optimize(sample_mu, cov_sample_full)

    weights_table = pd.DataFrame({
        "Asset": names,
        "Market (equilibrium input)": w_mkt,
        "Equilibrium optimal": w_eq,
        "Black-Litterman": w_bl,
        "Naive MV": w_naive,
    })
    print("\nFull-sample optimal weights:")
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(weights_table.to_string(index=False))
    weights_table.to_csv(os.path.join(outdir, "weights_comparison.csv"),
                         index=False)

    # ------------------------------------------------------------------
    # Step 4b: tracking-error-constrained variant and constraint-cost table.
    # ------------------------------------------------------------------
    w_6040_full = np.zeros(n)
    w_6040_full[names.index("US Large Cap Equity")] = 0.60
    w_6040_full[names.index("US Aggregate Bonds")] = 0.40
    cost_df = constraint_cost_table(mu_bl, cov_full, w_6040_full, cap=ASSET_CAP,
                                    te_budget=TE_BUDGET)
    print(f"\nConstraint-cost table (BL posterior mu, TE budget = "
          f"{TE_BUDGET:.0%} vs 60/40 benchmark):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
        print(cost_df.to_string(index=False))
    cost_df.to_csv(os.path.join(outdir, "constraint_cost_table.csv"),
                   index=False)

    # ------------------------------------------------------------------
    # Step 5: naive MV failure analysis (instability and concentration).
    # ------------------------------------------------------------------
    instab = bootstrap_instability(rets, w_mkt)
    enb_naive = effective_number_of_bets(w_naive)
    enb_bl = effective_number_of_bets(w_bl)
    print("\nNaive mean-variance failure analysis (bootstrap resampling):")
    print(f"  Avg L1 weight shift across resamples, naive MV: "
          f"{instab['naive_mean_L1']:.4f}")
    print(f"  Avg L1 weight shift across resamples, BL:       "
          f"{instab['bl_mean_L1']:.4f}")
    print(f"  Effective number of bets, naive MV: {enb_naive:.2f}")
    print(f"  Effective number of bets, BL:       {enb_bl:.2f}")
    stability_ok = instab["bl_mean_L1"] <= instab["naive_mean_L1"] + 1e-9
    print(f"  CHECK: BL is at least as stable as naive MV: "
          f"{'PASS' if stability_ok else 'FAIL'}")
    pd.DataFrame({
        "Model": ["Naive MV", "Black-Litterman"],
        "Mean L1 weight shift under resampling": [
            instab["naive_mean_L1"], instab["bl_mean_L1"]],
        "Effective number of bets": [enb_naive, enb_bl],
    }).to_csv(os.path.join(outdir, "naive_mv_failure_analysis.csv"),
              index=False)

    # ------------------------------------------------------------------
    # Step 6: quarterly walk-forward backtest from 2020 onward.
    # ------------------------------------------------------------------
    metrics_df, wealth_df, turnover_df, weights_log = run_walk_forward(rets, w_mkt)
    metrics_df.to_csv(os.path.join(outdir, "walkforward_metrics.csv"),
                      index=False)
    turnover_df.to_csv(os.path.join(outdir, "walkforward_turnover.csv"),
                       index=False)
    wealth_df.to_csv(os.path.join(outdir, "walkforward_wealth.csv"))

    # Turnover check: BL should trade less than, or comparably to, naive MV.
    avg_to = (turnover_df.groupby("Strategy")["Turnover"].mean())
    to_bl = avg_to.get("Black-Litterman", np.nan)
    to_naive = avg_to.get("Naive MV", np.nan)
    turnover_ok = to_bl <= to_naive + 1e-9
    print(f"\nAverage per-rebalance turnover: BL = {to_bl:.4f}, "
          f"naive MV = {to_naive:.4f}")
    print(f"CHECK: BL turnover <= naive MV turnover: "
          f"{'PASS' if turnover_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Step 7: tau and confidence sensitivity studies.
    # ------------------------------------------------------------------
    tau_df = tau_sensitivity(pi, cov_full, P, Q, confidence, w_mkt)
    conf_df = confidence_sensitivity(pi, cov_full, P, Q, confidence, w_mkt)
    print("\ntau sensitivity:")
    with pd.option_context("display.float_format", lambda v: f"{v:,.6f}"):
        print(tau_df.to_string(index=False))
    print("\nView confidence sensitivity:")
    with pd.option_context("display.float_format", lambda v: f"{v:,.6f}"):
        print(conf_df.to_string(index=False))
    tau_df.to_csv(os.path.join(outdir, "sensitivity_tau.csv"), index=False)
    conf_df.to_csv(os.path.join(outdir, "sensitivity_confidence.csv"),
                   index=False)

    # ------------------------------------------------------------------
    # Step 8: figures.
    # ------------------------------------------------------------------
    paths = [
        plot_wealth_paths(wealth_df, outdir, mode_label),
        plot_weight_comparison(names, w_mkt, w_bl, w_naive, outdir, mode_label),
        plot_expected_returns(names, pi, mu_bl, sample_mu, outdir, mode_label),
        plot_sensitivity(tau_df, "tau", ["L1 |mu_BL - pi|", "L1 |w_BL - w_mkt|"],
                         "Sensitivity of BL allocation to tau", outdir,
                         "sensitivity_tau.png"),
        plot_sensitivity(conf_df, "Confidence scale",
                         ["L1 |mu_BL - pi|", "L1 |w_BL - w_mkt|"],
                         "Sensitivity of BL allocation to view confidence",
                         outdir, "sensitivity_confidence.png"),
        plot_covariance_heatmap(cov_full, names, outdir, mode_label),
        plot_bl_weight_evolution(weights_log, names, outdir, mode_label),
    ]
    paths = [p for p in paths if p is not None]
    print("\nFigures written:")
    for p in paths:
        print(f"  {p}")
    print(f"\nAll CSV outputs written to: {outdir}")
    if args.mode == "synthetic":
        print("\nREMINDER: these results come from SYNTHETIC data and are a "
              "demonstration of method, not evidence about real markets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
