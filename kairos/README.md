# Kairos: a causal KAN that notices drift and can shape fairness

A causal graph whose equations are KAN curves: one curve per edge, summed at the
node, plus a surface for a pair of parents that interact.

    x_j = c_j + Σ φ_ij(x_i) + Σ ψ_abj(x_a, x_b) + u_j          (every variable)
    P(Y = 1) = σ(c_Y + Σ φ_iY(x_i) + Σ ψ_abY(x_a, x_b))         (the outcome)

## Files, in reading order

| File | Class | What it does |
|---|---|---|
| `src/causal_model/spline.py` | `Spline` | one curve: cubic B-spline bump weights @ heights, with a band |
| `src/causal_model/surface.py` | `Surface` | one interaction between two parents: a grid of heights holding only what depends on both |
| `src/causal_model/fitting.py` | `PlainFitting`, `BayesianFitting` | how heights are chosen: data only, or with a smoothness belief whose strength λ (one per curve, two per surface) is picked by the evidence |
| `src/causal_model/node.py` | `Node` | one variable: constant + curves + surfaces, fitted as least squares |
| `src/causal_model/outcome.py` | `OutcomeNode` | `Y`: the same, but a chance, fitted with damped Newton |
| `src/causal_model/screening.py` | `PairScreening` | which pairs of parents get a surface, decided from the data |
| `src/causal_model/kan.py` | `CausalKAN` | the graph: nodes in causal order |
| `src/causal_model/change.py` | `ChangeFit` | drift: one fit on people from before and after, with a change part per term that is kept only if the data needs it |
| `src/causal_model/fairness.py` | `FairnessPolicy`, `FairScore` | fairness: the decision with a protected attribute's influence removed (or reduced) along chosen edges |
| `src/causal_model/plot.py` | `KANPlot` | one panel per curve, one heat map per surface |
| `src/cf_generation/counterfactual.py` | `Counterfactual` | abduction, action, prediction |

Simulations, with worlds whose true equations are known:

| File | What it does |
|---|---|
| `src/simulation/world.py` | `World` and its mechanisms (`Uniform`, `Binary`, `Equation`, `Outcome`) |
| `src/simulation/scenarios.py` | six worlds, each with one drift |
| `src/simulation/drift.py` | runs each world once and saves figures and a summary in `figures/worlds/<world>/` |
| `src/simulation/repeat.py` | runs worlds over many seeds and counts how often each term is flagged |
| `src/simulation/fairness_demo.py` | a world with a protected group; fair scores checked against the truth, in `figures/fairness/` |
| `src/simulator.py` | the first small world, fitted plain and Bayesian side by side |

## Run (from `kairos/`)

    PYTHONPATH=. venv/bin/python src/simulation/drift.py
    PYTHONPATH=. venv/bin/python src/simulation/repeat.py --worlds 6 --seeds 5 --figures
    PYTHONPATH=. venv/bin/python src/simulation/fairness_demo.py

## Left out on purpose

- Batch only: no running statistics, forgetting or change windows over time yet.
- Knots are evenly spaced over the data's range; curves are flat outside it.
- Yes/no variables are fitted as curves between 0 and 1; only the two ends mean anything.
- Fairness can only shape edges out of the protected attribute, and it shapes the
  decision score, not recourse (there is no recourse search yet).
- For `Y`, a counterfactual reports the chance `P(Y = 1)`, not a 0/1 value.
