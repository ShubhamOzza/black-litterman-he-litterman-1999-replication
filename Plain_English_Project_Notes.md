# Plain English Project Notes: Black-Litterman Portfolio Allocation

## What we made

We built a computer program that decides how to split money across ten different kinds of investments (US stocks, international stocks, bonds, real estate, gold, and so on). The program uses a famous method called the Black-Litterman model, which big asset managers like BlackRock have used for decades. The program also proves, with automatic tests and a simulated race against simpler strategies, why this method is better than the naive textbook approach.

## The concept in plain words

Imagine you want to build a portfolio and you ask a computer to find the best mix. The textbook approach tells the computer: look at what each investment earned on average in the past, and pick the mix with the best expected return for a given amount of risk. The problem is that past averages are extremely noisy estimates of the future. The optimizer does not know this; it treats those noisy averages as exact truth. So it piles money into whatever happened to do well in your data sample, which is often just luck. Change the sample slightly and the "optimal" portfolio changes completely. Quant people call this behavior an error maximization machine.

Black-Litterman fixes this with two ideas:

1. Start from what the market already believes. If millions of investors collectively hold a certain mix of assets, that mix must reflect a set of reasonable expected returns. We can run the math backward: given the market's portfolio and the way assets move together, what expected returns would make that portfolio the smart choice? These are called equilibrium returns, and they are stable and sensible by construction.
2. Let the investor nudge the answer with opinions. You might think, for example, that US large companies will beat international ones by 2 percent a year, but you are only moderately sure. Black-Litterman blends your views with the market equilibrium using Bayesian statistics, and each view carries a confidence dial. Zero confidence means the view is ignored completely; total confidence means the view is treated as fact. Most of the time you are somewhere in between, and the model handles that smoothly.

The result is a portfolio that stays close to the diversified market mix unless your views are strong, and that does not thrash around when the data wiggles.

## How we made it, step by step

1. Built a dataset of monthly returns for ten asset classes from 2015 to 2025. Because live market data is not always available, the program can run on a clearly labeled synthetic dataset that behaves like real markets, and it can also read a real CSV file or download real data from Yahoo Finance.
2. Estimated how the assets move together using a shrinkage technique (Ledoit-Wolf) that deliberately damps statistical noise in correlations.
3. Ran the backward calculation to get equilibrium expected returns from market weights.
4. Added three realistic economic views (one about US versus international stocks, one about bonds, one about emerging markets), each with its own confidence level.
5. Computed the Black-Litterman posterior, which is the mathematically exact blend of equilibrium and views.
6. Optimized the final portfolio under real-world rules: all money invested, no shorting, and no single asset above 25 percent. We also built a second version that adds a tracking-error budget against a 60/40 benchmark, and a table showing how much return-generating power each rule set gives up compared with an unconstrained portfolio (in this case, almost nothing).
7. Proved the math is right with unit tests: with zero-confidence views the model must give back equilibrium exactly, and with total-confidence views it must hit the views exactly. Both hold to machine precision.
8. Showed why the naive approach fails: when we resample the data with bootstrapping, the naive portfolio changes by about 61 percent of its value from pure noise, while the Black-Litterman portfolio changes by about 3 percent.
9. Ran a realistic walk-forward race: every quarter from 2020 onward, each strategy is rebuilt using only the previous five years of data, then judged on what happened next. Black-Litterman traded several times less than the naive optimizer and posted a better risk-adjusted return (Sharpe ratio) and a smaller drawdown, even though the naive portfolio's raw growth rate was a touch higher by luck in this particular simulated run. We honestly report both facts, and that a simple 60/40 stock-bond mix remains a tough benchmark.
10. Mapped the sensitivity of the answer to the two main control dials (tau and view confidence), confirming the model behaves exactly the way the theory says it should.

## Interview-ready explanation

If an interviewer asks "tell me about this project", a strong 60-second answer is:

"I implemented the Black-Litterman model end to end in Python. Naive mean-variance optimization is unusable in practice because it amplifies estimation error in sample means into concentrated, unstable portfolios. Black-Litterman fixes this by anchoring expected returns to the equilibrium implied by the market portfolio through inverse optimization, pi equals delta times Sigma times w, and then blending in investor views with a Bayesian posterior where each view has its own uncertainty. I used Ledoit-Wolf shrinkage for the covariance matrix and solved the constrained long-only quadratic program, plus a tracking-error-constrained variant, with cvxpy when it is available and a scipy SLSQP fallback otherwise; in my own test environment cvxpy was not installed, so I made sure the report is explicit that the SLSQP path is what actually produced every number. I verified the model's limiting cases with unit tests: zero-confidence views recover equilibrium to machine precision, and infinite-confidence views satisfy the view equations exactly. I also quantified the naive failure mode with bootstrap resampling, where naive weights moved about eighteen times more than Black-Litterman weights, and ran a quarterly walk-forward backtest from 2020 onward against 60/40 and equal-weight benchmarks. Black-Litterman cut turnover to roughly a sixth of naive optimization's and won on Sharpe ratio, volatility, and drawdown, though naive MV's raw CAGR was marginally higher in this particular sample, which I report rather than hide. One thing I am careful about: I do not claim Black-Litterman beats every benchmark out of sample, because honest backtests usually show that simple diversified portfolios are hard to beat, and overfitting is the biggest risk in this kind of work."

Likely follow-up questions and short answers:

- What is tau? It scales how uncertain we are about the equilibrium prior. Small tau means we trust equilibrium strongly and views move the answer very little; large tau lets the views dominate. The report includes a sensitivity study showing the effect is smooth and monotonic.
- What is Omega? The uncertainty matrix of the views. We set each view's uncertainty proportional to the risk of the assets involved in that view, divided by our confidence, following He and Litterman.
- Why shrink the covariance? Sample correlations are noisy; shrinkage trades a small bias for a large reduction in variance, which is the right trade when the optimizer will exploit any noise it finds.
- How do you know it is not overfit? Trailing-window walk-forward testing with no tuning on the test period, fixed rules, sensitivity studies that expose the tuning knobs, and unit tests on theoretical identities rather than on backtest performance.
- What would you add in production? Transaction costs and taxes, time-varying market weights, refreshed views each quarter, and uncertainty in the covariance matrix itself.
