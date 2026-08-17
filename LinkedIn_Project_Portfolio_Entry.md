# LinkedIn Project Portfolio Entry

## Project Title

Black-Litterman Multi-Asset Portfolio Allocation: Bayesian Blending of Market Equilibrium and Investor Views (Python)

## Compelling Description

Naive mean-variance optimization has a dirty secret: feed it raw historical returns and it becomes an error maximization machine, producing concentrated portfolios that swing wildly every time the data window moves. I built a complete, unit-tested Black-Litterman allocation engine in Python to show how professionals fix this.

The engine reverse engineers equilibrium expected returns from the market portfolio (pi = delta * Sigma * w_mkt), blends them with explicit economic views through a Bayesian posterior (He-Litterman view uncertainty, tau = 0.05), and solves a constrained long-only quadratic program with a 25 percent per-asset cap, plus a tracking-error-constrained variant, using cvxpy when available and a scipy SLSQP fallback otherwise (the fallback is what actually ran in my own test environment, and the report says so explicitly).

What makes this project different from a typical notebook:

- Correctness is proven, not assumed. A six-test suite verifies the model's theoretical limiting cases to machine precision: zero-confidence views recover equilibrium exactly, infinite-confidence views satisfy P * mu = Q exactly, and covariance matrices stay positive semidefinite.
- The failure of naive MV is quantified, not just asserted. Bootstrap resampling shows naive weights move by 61 percent of the portfolio in response to pure sampling noise, versus 3 percent for Black-Litterman, and naive MV pins three separate positions to the 25 percent cap while Black-Litterman stays diversified across all ten assets.
- Out-of-sample honesty. A quarterly walk-forward backtest from 2020 onward compares BL against 60/40 and equal-weight benchmarks with strictly trailing estimation windows. BL cut average turnover to roughly a sixth of naive MV's and posted a better Sharpe ratio, lower volatility, and a shallower drawdown, while naive MV's raw CAGR was marginally higher in this sample, a result the report states plainly rather than hides. The 60/40 benchmark remains the strongest Sharpe ratio in the sample.
- A constraint-cost table shows what each rule actually costs: moving from a fully unconstrained solution to the long-only, capped, tracking-error-budgeted portfolio gives up roughly 0.001 percent of quadratic utility, evidence that the constraint set restrains the unstable naive optimizer without materially distorting an already well-anchored one.
- Full reproducibility: one command runs the whole pipeline on a labeled synthetic dataset, and the same script accepts real CSV data or live yfinance downloads with a single flag.

Full disclosure: every result, figure, and backtest statistic shown in this project comes from the clearly labeled synthetic demonstration mode and is not live-market evidence; the identical pipeline is ready to run on real data supplied by the user.

## Skills (comma separated)

Black-Litterman Model, Portfolio Construction, Mean-Variance Optimization, Bayesian Inference, Inverse Optimization, Covariance Shrinkage, Ledoit-Wolf Estimation, Quadratic Programming, cvxpy, scipy Optimization, Walk-Forward Backtesting, Turnover Analysis, Sensitivity Analysis, Estimation Error, Risk Modeling, Multi-Asset Investing, Asset Allocation, Python, NumPy, pandas, Matplotlib, Unit Testing, Reproducible Research

## Media Recommendations

1. Hero image: the walk-forward cumulative wealth chart, showing BL, naive MV, 60/40, and equal weight over 2020 to 2025. It communicates both competence and honesty at a glance.
2. The weights comparison bar chart (equilibrium vs BL vs naive MV), which makes the corner-solution problem instantly visible.
3. The Black-Litterman weight evolution stacked area chart, showing how the allocation adapts smoothly across 24 quarterly rebalances.
4. The tau and confidence sensitivity line charts, evidence that the model behaves exactly as theory predicts at its limits.
5. A short screen recording (30 to 60 seconds) of the --selftest run printing six PASS lines, a strong signal of engineering rigor for hiring managers.
6. Link out to the GitHub repository containing the script, README, and academic report.
