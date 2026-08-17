# Black-Litterman Portfolio Allocation

Project I-8 of the Quant Master series (Portfolio Construction). A complete, unit-tested implementation of the Black-Litterman model for multi-asset allocation, including inverse optimization of equilibrium returns, shrinkage covariance estimation, Bayesian view blending, constrained mean-variance optimization, naive MV failure analysis, quarterly walk-forward backtesting, and sensitivity studies.

## Quick start

```bash
pip install -r requirements.txt

# Run the full pipeline on the clearly labeled synthetic demo dataset
python Black_Litterman_Portfolio_Allocation.py --mode synthetic

# Run only the correctness and limiting-case test suite
python Black_Litterman_Portfolio_Allocation.py --selftest

# Use your own monthly returns CSV
python Black_Litterman_Portfolio_Allocation.py --mode csv --csv my_returns.csv

# Use your own CSV of adjusted monthly prices (converted to returns)
python Black_Litterman_Portfolio_Allocation.py --mode csv --csv my_prices.csv --prices

# Download real data with yfinance (requires internet access)
python Black_Litterman_Portfolio_Allocation.py --mode yfinance
```

All figures and CSV tables are written to `outputs/` (override with `--outdir PATH`).

## File map

| File | Contents |
|---|---|
| `Black_Litterman_Portfolio_Allocation.py` | The complete pipeline: data layer, equilibrium, BL posterior, optimizers, unit tests, walk-forward backtest, sensitivity studies, plots, CSV exports |
| `Word_Document_Content.md` | Academic report content: abstract, methodology, application, interpretation, validation, limitations, conclusion, references |
| `LinkedIn_Project_Portfolio_Entry.md` | Portfolio-ready title, description, skills list, media recommendations |
| `Plain_English_Project_Notes.md` | Jargon-free explanation plus an interview-ready summary |
| `README.md` | This file |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated figures (PNG) and tables (CSV); created on first run |

## Data schema

Input data, in any mode, is a panel of monthly simple returns:

- Index: month-end dates.
- Columns: between 8 and 12 assets.
- Values: simple monthly returns as decimals (0.02 means plus 2 percent).

CSV mode expects the first column to be a parseable date and the remaining columns to be numeric. With `--prices`, the same layout is interpreted as adjusted prices and converted to returns by percentage change. Synthetic mode generates 132 monthly observations (January 2015 through December 2025) for ten asset-class proxies calibrated to plausible means, volatilities, and correlations, with a fixed random seed.

## Dependencies

Required: numpy, pandas, scipy, matplotlib. Strongly recommended: scikit-learn (Ledoit-Wolf shrinkage; a closed-form fallback exists), cvxpy (primary constrained optimizer; a scipy SLSQP fallback exists). Optional: yfinance (live data mode only). Both optional packages degrade gracefully and the script prints which backend actually ran (`Solver backend: cvxpy` or `Solver backend: scipy SLSQP fallback`) so results are never silently mislabeled.

## Model parameters

- delta (risk aversion) = 2.5, used in pi = delta * Sigma * w_mkt.
- tau (prior uncertainty scale) = 0.05, tunable over roughly 0.01 to 0.25 (see `sensitivity_tau.csv`).
- View uncertainty: Omega_k = (P Sigma P^T)_kk / c_k, so Omega is proportional to P Sigma P^T through the confidence channel; tau is kept out of Omega so the two controls remain separately identified. Confidence scaling is tunable (see `sensitivity_confidence.csv`).
- Constraints: weights sum to one, long-only, per-asset cap 25 percent, plus an optional tracking-error-constrained variant (5 percent annualized TE budget versus the 60/40 benchmark). All three constraint tiers (unconstrained, capped, TE-constrained) are compared in `constraint_cost_table.csv`.
- Walk-forward: 60-month trailing estimation window, quarterly rebalancing, out-of-sample from January 2020.

## Notes and honesty statements

- The synthetic demo mode exists to exercise and validate the full workflow without market data. Its outputs are labeled SYNTHETIC in file names, figure titles, and console output, and they must never be presented as real market evidence.
- In csv and yfinance modes, supply your own market-cap weights for production-grade equilibrium priors; the script otherwise uses equal weights with an explicit warning.
- The demo deliberately does not tune parameters on the out-of-sample period. Walk-forward results are reported as they come out, including where simple benchmarks remain competitive and where naive mean-variance's raw CAGR is not beaten by Black-Litterman.
- Solver honesty: the constrained optimizer prefers cvxpy and falls back to scipy SLSQP when cvxpy is not installed. In the reference environment this project was verified in, cvxpy was not installed, so every number in the reports and figures was produced by the scipy SLSQP fallback path, not cvxpy. scikit-learn was installed, so the Ledoit-Wolf shrinkage path (not its closed-form fallback) is what actually ran.

## Verification

- `python -m py_compile Black_Litterman_Portfolio_Allocation.py` passes.
- `--selftest` runs six unit tests (BL limiting cases, equilibrium plausibility, optimizer constraints, positive semidefiniteness), all passing.
- The synthetic demo run completes end to end and writes all figures and CSVs to `outputs/`, and re-running it reproduces byte-identical CSV outputs (fixed seed).
