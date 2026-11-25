# Known issues of kairos' causal discovery

Found while benchmarking on CausalArena (November 2026). Both come from the same
assumption: every variable is scored as

    x = constant + Σ curves(parents) + bell-shaped (Gaussian) noise

and the score of a variable (`Node._solve`, `src/causal_model/nodes/node.py`) is
driven by one term, −N/2 · log σ̂², where σ̂² is the variance left after the fit.

| # | Issue | Which side of an arrow | Main cost |
|---|---|---|---|
| 1 | [The noise model is wrong](01-noise-model.md) | the child | wrong arrow directions, inflated scores |
| 2 | [The input model is wrong for categories](02-categorical-inputs.md) | the parent | fits that depend on arbitrary codes |

Evidence: on the mortgage scenario (rep 0), kairos' F1 is 0.24 with no background
and 0.48 when the roots are given.
