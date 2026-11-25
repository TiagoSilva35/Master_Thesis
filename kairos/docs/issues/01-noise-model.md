# Issue 1: the noise model is wrong

## The problem

The score assumes the leftover of every variable is bell-shaped, so it only looks
at the leftover's **size** (σ̂²), never its **shape**. Three cases break this.

### 1a. Non-bell roots and noise → wrong directions

A root has no parents, so it is all noise; its whole distribution is judged as a
bell curve. Comparing X → Y with Y → X then becomes "which wrong model looks less
wrong", and the reversed arrow can win.

**Example (straight line).** Y = 0.8·X + noise. With straight lines, X → Y and
Y → X get exactly the same score on any data: variance alone cannot tell cause
from effect.

**Example (uniform cause).** X is uniform on [0, 1] (every value equally likely),
Y = X² + small noise.
- X → Y: X is scored as a bell curve, which fits a flat distribution badly.
- Y → X: X ≈ √Y fits almost perfectly, so X's leftover is tiny; now Y is scored
  as a bell curve instead.

Which total is larger depends on how badly each variable violates the bell
shape, not on which is the cause. Often the reverse wins.

**In CausalArena.** Roots are drawn from truncated normals, log-normals,
categories and Bernoulli draws. On the mortgage scenario F1 halves (0.48 → 0.24)
when kairos is not told the roots.

### 1b. Categorical children → leftovers that cannot be bell-shaped

**Example.** `refinance_application_attempt` is 0 or 1. If the curve predicts 0.3
for a borrower, the leftover is −0.3 or +0.7 and nothing in between. σ̂² no
longer measures the fit; the right model is a probability,
P(attempt = 1) = sigmoid(curve). kairos does this for the outcome Y
(`OutcomeNode`), not for other binary variables.

Same for nominal children (`observed_event_type_code`: none, refinance, sale,
default) and counts (`remaining_term`).

### 1c. Near-zero noise → scores that explode and no direction

When a variable is an almost exact formula of its parents, σ̂² → 0 and
−N/2 · log σ̂² → ∞.

**Example (the spread).** `current_rate_spread` = note rate − refinance index,
rounded to 0.01 percentage points (e.g. 6.25 − 4.80 = 1.45). Its only noise is
the rounding. With N = 1,000:

| Child | Points |
|---|---|
| a root (σ̂² = 1) | 0 |
| a typical child (σ̂² = 0.2) | ≈ 800 |
| spread ← note, index | 6,286 |
| note ← spread, index | 6,050 |
| index ← note, spread | 5,811 |

- **Too big:** one arrow is worth eight ordinary ones, so the greedy search takes
  it first and, if it is reversed, blocks right arrows later.
- **No direction:** the formula can be rearranged and every arrangement leaves the
  same rounding error. The true one wins only because the spread has the largest
  raw variance (≈ 1.5 + 0.9), and standardising divides the same error by it.

**Example (exact tie).** recorded month = true month + a 0–2 month lag. Both
directions score 3,678: with one parent and a straight line, both leave the same
share of variance unexplained.

**Counter-example (safe).** `right_censored_at_horizon` = 1 if no event by month
60. From the month you know the flag; from "flag = 0" you cannot recover the
month. Formulas that lose information are near-exact in the true direction only.

## Possible remedies

| Case | Remedy |
|---|---|
| 1a | Gaussianise marginals (rank/quantile transform); a residual-independence score (RESIT/HSIC); a flexible noise density |
| 1b | logistic (binary), softmax (nominal), ordered logit or Poisson (ordinal, counts) |
| 1c | a prior on σ² (turns log σ̂² into log(σ̂² + c), a soft floor); declare computed columns as known formulas; orient by information loss or interventions |
