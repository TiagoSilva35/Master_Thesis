# Kairos: a causal KAN that notices drift

A causal graph whose equations are KAN curves: one curve per edge, summed at the
node, plus a surface for a pair of parents that interact.

    x_j = c_j + Σ φ_ij(x_i) + Σ ψ_abj(x_a, x_b) + u_j          (every variable)
    P(Y = 1) = σ(c_Y + Σ φ_iY(x_i) + Σ ψ_abY(x_a, x_b))         (the outcome)

## Files, in reading order

| File | Class | What it does |
|---|---|---|
| `src/causal_model/spline.py` | `Spline` | one curve: cubic B-spline bump weights @ heights, with a band; beyond the data it continues as a straight line |
| `src/causal_model/surface.py` | `Surface` | one interaction between two parents: a grid of heights holding only what depends on both |
| `src/causal_model/fitting.py` | `PlainFitting`, `BayesianFitting` | how heights are chosen: data only, or with a smoothness belief whose strength λ (one per curve; three per surface: along each parent and on the saddle) is picked by the evidence |
| `src/causal_model/node.py` | `Node` | one variable: constant + curves + surfaces, fitted as least squares |
| `src/causal_model/outcome.py` | `OutcomeNode` | `Y`: the same, but a chance, fitted with damped Newton |
| `src/causal_model/screening.py` | `PairScreening` | which pairs of parents get a surface, decided from the data |
| `src/causal_model/kan.py` | `CausalKAN` | the graph: nodes in causal order |
| `src/causal_model/change.py` | `ChangeFit` | drift: one fit on people from before and after, with a change part per term that is kept only if the data needs it |
| `src/causal_model/plot.py` | `KANPlot` | one panel per curve, one heat map per surface |

Simulations, with worlds whose true equations are known:

| File | What it does |
|---|---|
| `src/simulation/world.py` | `World` and its mechanisms (`Uniform`, `Binary`, `Equation`, `Outcome`) |
| `src/simulation/scenarios.py` | six worlds, each with one drift |
| `src/simulation/drift.py` | runs each world once and saves figures and a summary in `figures/worlds/<world>/` |
| `src/simulation/repeat.py` | runs worlds over many seeds and counts how often each term is flagged |
| `src/simulator.py` | the first small world, fitted plain and Bayesian side by side |

## Run (from `kairos/`)

    PYTHONPATH=. venv/bin/python src/simulation/drift.py
    PYTHONPATH=. venv/bin/python src/simulation/repeat.py --worlds 6 --seeds 5 --figures

## Tests (from `kairos/`)

    venv/bin/python -m pytest

Written as behaviour scenarios (Given / When / Then) in `tests/features/*.feature`,
with the steps in `tests/test_*.py` (pytest-bdd). They check the model against
independent references: the textbook B-spline formula, the direct evidence formula,
importance sampling for Y's evidence, and world 1's true equations for band coverage
and predictions past the data. A scenario tagged `@known_problem` describes behaviour
the model does not have yet: it is expected to fail, and pytest reports when it starts
passing. None is tagged at the moment.

## Left out on purpose

- Batch only: no running statistics, forgetting or change windows over time yet.
- Knots are evenly spaced over the data's range. Beyond it a curve continues as a
  straight line with its slope at the edge, and its band widens; `Spline.outside(x)`
  tells when a value is past the data.
- Yes/no variables are fitted as curves between 0 and 1; only the two ends mean anything.
- No fairness policy: the causal model only describes the world as it is. A
  protected attribute can be a variable, and the curves on its edges show where
  a gap enters (with bands), but which routes are unfair is decided outside the
  model, in the recourse search and the monitor, which are not built yet.
- No counterfactuals for now: the code that answered "what if this person's income
  changed?" was removed and will return with the recourse search.
