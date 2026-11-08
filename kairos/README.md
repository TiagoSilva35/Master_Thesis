# Causal KAN: foundations

A causal graph whose equations are KAN curves: one curve per edge, summed at the node.

    x_j = c_j + Σ φ_ij(x_i) + u_j          (every variable)
    P(Y = 1) = σ(c_Y + Σ φ_iY(x_i))        (the outcome)

The foundations are done when the model can say how a person's chance of a good
outcome changes if their income changes. Read the files in this order:

| Step | File | Class | What it adds |
|---|---|---|---|
| 1 | `src/causal_model/spline.py` | `Spline` | one curve: cubic B-spline weights @ heights |
| 2 | `src/causal_model/node.py` | `Node` | one variable: constant + one curve per parent, fitted by least squares |
| 3 | `src/causal_model/kan.py` | `CausalKAN` | the graph: nodes in causal order, each fitted on its own |
| 4 | `src/counterfactual.py` | `Counterfactual` | abduction, action, prediction |
| 5 | `src/outcome.py` | `OutcomeNode` | `Y` with a logistic equation |

`src/toy_world.py` is a small world with known equations (income → savings, both → Y)
that the tests and the demo check the model against.

Run from the repo root:

    python3 -B -m unittest discover -s kairos/tests -t .
    python3 -B -m kairos.demo

## Left out on purpose

- Knots are evenly spaced over the training range; curves are flat outside it,
  so the slope jumps to zero at the first and last knot.
- A parent with a single value (for example a binary protected attribute) raises
  `ValueError`, because its knots would all sit at one point.
- No online updates, drift detection, recourse search, or fairness yet.
- For `Y`, a counterfactual reports the chance `P(Y = 1)`, not a 0/1 value:
  `Y` is not known for the person being asked about, so its noise is not recovered.
- Newton runs a fixed number of steps with no convergence check.
