# Measurement types and noise: plan of action

> **25 November 2026: partly superseded.** Phases 0, 2, 3 and 5 (kinds, categorical children and
> parents, interventional rows) are still how kairos works. Phase 1 (carried noise floor), Phase 4
> (learnt density and spread curves) and Phase 6 (order search) were replaced by simpler parts; their
> code is in `diagnostics/set_aside/`. `HANDOVER.md` says what replaced them, and why.

## 1. The problems

kairos scores every variable with one model:

  x = constant + Σ curve(parent) + bell-shaped noise

and the evidence mostly rewards a small leftover variance σ̂² (the term −N/2 · log σ̂² in
`Node._solve`). CausalArena's variables break this model in four ways. Figures are from the
mortgage scenario (24 variables, 1,000 rows, measured 2026-11-18).

| # | Problem | Example | What it costs |
|---|---|---|---|
| P1 | Near-zero noise | spread = note rate − index, rounded to 0.01: σ̂² = 0.0000035, worth 6,300 points. Its two rearrangements score 6,050 and 5,811. | One arrow outweighs about 8 ordinary ones, and the greedy search takes it first. Its direction follows the variables' raw variances, not causation. |
| P2 | Categorical children | 3 binary, 2 nominal and 3 ordinal variables with few levels are scored as bell-shaped | The likelihood is wrong. For nominal variables, the result depends on how the categories happen to be numbered. |
| P3 | Categorical parents | event type (none, refinance, sale, default) enters a spline as 0, 1, 2, 3 | The order is made up, and an 8-interval spline is spread over 2–5 values. |
| P4 | Bell-shaped noise and roots | roots that are log-normal, truncated or categorical | The score cannot tell the direction: F1 is 0.24 without background and 0.48 with the roots given. |

There is also one missed opportunity (P5). In the obs_int files, rows 800–999 each set one
variable by intervention. kairos fits those rows as if nothing had happened.

## 2. Words used below

| Word | Meaning | Example |
|---|---|---|
| Likelihood | How probable the data is under a model with given numbers. For a continuous variable it is a density, for a categorical one a probability. | P(attempt = 1) = 0.3 for one borrower |
| Laplace approximation | When the evidence has no formula: climb to the top of the likelihood hill (Newton steps) and treat the hill as a bell there. `OutcomeNode` does this today. | |
| Logistic model | A 0/1 child: P(x = 1) = σ(curves), with σ(s) = 1 / (1 + e^−s) | `OutcomeNode` |
| Ordered logit | An ordinal child with k levels: P(x ≤ level c) = σ(θ_c − curves). There is one set of curves and k − 1 cut-points θ in increasing order. | credit band 0–4 |
| Softmax | A nominal child: P(x = c) is proportional to e^(curves_c). There is one set of curves per category, except one reference category. | event type |
| Prior | A belief held before seeing the data | "noise is rarely below 1% of a variable's variance" |
| Entropy | How spread out a distribution is, in log units. Lower entropy means more predictable. | |
| Intervention | Setting a variable from outside, so that its own parents no longer drive it | an experiment changes the subsidy |
| Log-odds | log(p / (1 − p)): a probability stretched onto the whole number line | p = 0.5 → 0, p = 0.95 → 2.9, p = 0.99995 → 10 |
| Concave | Bends downward everywhere, like an upside-down bowl, so it has at most one top | −h² |

## 3. Ground rules

- **The defaults keep today's behaviour:** every variable continuous, no floor, bell-shaped
  noise. The simulations, the drift code and the 65 existing tests run unchanged. The
  benchmark wrapper switches the new behaviour on.
- **Each phase ends with:** all tests passing, pyright at 0 errors, and a smoke test.
- **Smoke tests only:** mortgage rep 0 in both regimes, plus one scenario from another people
  domain so the work is not tuned to one file. The data goes into the scratchpad (about 1 MB),
  and I ask before downloading. Never a full run; that is yours.
- **No commits.** After each phase I list the changed files so you can use Compare.
- **Code follows the python-development skill:** small classes, one job each, no new globals.

## 4. Phases

The order matters. Phase 1 is small and stops the worst distortion at once. Phase 2 comes
before Phase 4, so the learnt noise shape is never fitted to 0/1 residuals.

### Phase 0: types reach kairos (no change in behaviour)

**What:**
- `Background` gets `kinds: dict[str, str]`, with values "continuous", "binary", "ordinal" or
  "nominal". A missing name means continuous. A variable's type is known before looking at
  the data, which is what `Background` holds.
- The benchmark wrapper (`methods/kairos.py`) maps CausalArena's measurement scales to kinds:

  | CausalArena scale | kairos kind |
  |---|---|
  | continuous, positive_continuous, probability | continuous |
  | binary | binary |
  | categorical_nominal | nominal |
  | ordinal_categorical, ordinal_integer, discrete_ordinal, discrete_count | ordinal up to 10 levels, else continuous (D2) |

  The mortgage scenario should come out as about 16 continuous, 3 ordinal, 3 binary and
  2 nominal (the event months have 60 levels, so they count as continuous).
- The scales are read from the scenario's graph file (`ArenaData._kinds`), but only each
  variable's `measurement_scale`, never its arrows, so they reveal nothing about the answer.
- Check on the downloaded files that categorical codes are integers 0…k−1. Also check which
  variables pile up at a bound: the default hazard has 584 distinct values in 1,000 rows,
  probably clipped.
- Record the "before" smoke numbers.

**Tests:** the toy repository's scales (continuous, binary, ordinal_categorical) map to
continuous, binary and ordinal.

**Done when:** the kinds arrive in `GraphSearch` and nothing else changes.

### Phase 1: a soft floor on the noise (P1)

**What:**
- `Node(noise_floor=0.0)` replaces σ̂² with σ̂² + c wherever σ̂² appears in `_solve`, where
  c = noise_floor × the child's variance (carried through the equation since 2026-11-24, see below).
- This is the evidence you get when σ² has a prior instead of being estimated from the data
  (an inverse-gamma prior, the usual one for a variance). So the score stays a Bayesian
  evidence. It changes little while σ̂² is well above c, and it stops the score running off
  to infinity as σ̂² → 0.
- `GraphSearch` and `EvidenceScore` pass the floor through; the benchmark sets 0.01 (D1; 0.0001 since
  2026-11-24, see below).
- No floor (0) gives exactly today's evidence.

**What it does not fix:** direction. The rearrangements of a formula all reach the cap and
tie, so the existing doubtful flag (step 3) catches them. That is the honest answer for
reversible formulas when only observational data are available.

**Tests** (`noise_floor.feature`):
- A variable that is a rounded formula of its parents gets a gain of at most
  N/2 · log(1/0.01) plus a little, while without the floor it gets more.
- The floor changes the evidence of an ordinary noisy relation by less than 1%.
- With a floor of 0, `test_evidence` passes unchanged.

**Smoke:** the spread's arrows score much lower and are flagged doubtful. Count the doubtful
arrows.

#### The floor carried through the equation (added 2026-11-24)

**Why.** c = 1% of the *child's* variance gives the same formula a different floor depending on
which variable it is solved for, so the rearrangements do not tie. Take x and y independent
with variance 1, z = x − y exactly (variance 2), N = 1,000, and leave the prices aside:

| | True graph: x → z ← y | Wrong graph: z → y, then y and z → x |
|---|---|---|
| Explained normally | x: 0, y: 0 | z: −500·log 2 = −347; y given z: −500·log ½ = +347 |
| The formula's child | z: −500·log(1% of 2) = 1,956 | x: −500·log(1% of 1) = 2,303 |
| Total | 1,956 | 2,303 |

The wrong graph wins by 347 because x varies less than z.

**What.** Every continuous *variable* is trusted only up to a share of its own variance, and an
equation inherits the imprecision of everything in it (propagation of measurement error):

    floor = share × var(child) + Σ slope² × share × var(parent)

In the toy the floor is 0.01 + 0.01 + 0.02 = 0.04 whichever way the formula is written. Both
graphs score 1,609, and the true one wins because it needs one curve fewer.

- **`NoiseFloor`** (new `nodes/noise_floor.py`) holds the share and, per continuous parent, its
  imprecision and the matrix that turns the coefficients into slopes. `Node` builds it when it
  is fitted (`carried_floor`); a bell-shaped node uses the average over the rows, a
  `FlexibleNode` each row's own. Categorical parents are taken as recorded exactly.
- **The share is now 0.0001** (each variable is trusted up to 1% of its spread; 0.01 meant 10%).
  It was chosen on synthetic pairs y = f(x) + noise (5 formulas, 2 shapes, 5 seeds), never on
  the benchmark. Directions the learnt noise gets right:

  | Share | noise 10% of spread | noise 20% | noise 40% |
  |---|---|---|---|
  | 0.01, child only (before) | 34% | 86% | 94% |
  | 0.01, carried | 30% | 28% | 74% |
  | 0.001, carried | 68% | 92% | 94% |
  | 0.0001, carried | 82% | 92% | 94% |
  | 0.00001, carried | 80% | 92% | 94% |

  A floor near the size of the real noise blurs what tells the directions apart. The carrying is
  what fixes the formulas, at any share: with the child's share alone and learnt noise, the
  wrong graph of z = 100x − 30y stays 800–1,190 ahead at every share from 0.01 to 0.000001.

**What it does not fix:**
- A straight-line formula of bell-shaped variables is nearly a tie: only the extra curve counts.

**Tests** (`noise_floor.feature`): the floor is the child's share plus the carried ones; the same
formula has the same floor whichever variable it is solved for; a categorical parent adds
nothing; on z = x − y and z = 100x − 30y the wrong graph wins by more than 200 with the child's
share alone (0.01), by at most 20 carried (0.01) and by at most 3 carried (0.0001); the direction of ordinary pairs is kept; slopes match
small steps.

### Phase 2: categorical children (P2)

**What:**
- **`LaplaceNode(Node)`** (new `nodes/laplace.py`): the Newton climb and Laplace evidence, moved
  out of `OutcomeNode` unchanged. Each subclass provides only its log-likelihood, slope and
  curvature.
  - **`BinaryNode(LaplaceNode)`**: the logistic model, which is what `OutcomeNode` does now.
    `OutcomeNode(BinaryNode)` keeps only the outcome-specific parts (no children,
    `noise()` = 0).
  - **`OrdinalNode(LaplaceNode)`**: ordered logit, with one set of curves plus k − 1 ordered
    cut-points.
  - **`CategoryNode(LaplaceNode)`**: softmax, with one set of curves per category except the
    reference. One λ per parent is shared by all categories (D3).
- **Choosing the class:** `EvidenceScore` and `DiscoveredGraph.build` pick the node class from
  the kind. Today they pick `OutcomeNode` for the outcome and `Node` for everything else.
- **Values:** the benchmark wrapper passes categorical children their raw codes, not the
  standardised values.
- **Background docstring:** its advice to make yes/no variables roots, because "a bell-shaped
  Node would score poorly as a child", is no longer needed and is removed.
- **Counterfactuals:** the noise of a categorical child is not defined, so `noise()` returns 0,
  as `OutcomeNode` does. Nothing calls `noise()` today; counterfactuals for categorical
  variables are out of scope.
- **One hill, so Newton cannot settle on the wrong one.** The Laplace approximation puts a bell
  on the hill that Newton climbs, so a second hill would be missed. Here there is no second
  hill:
  - the log-likelihoods of the logistic, ordered logit and softmax models are concave in the
    curves' values (Pratt 1981 shows it for ordered models);
  - the curves are weighted sums of the heights;
  - the penalty is an upside-down bowl.

  A sum of concave pieces is concave, so it has exactly one top, and Newton reaches it from
  any start. The search over λ is a separate step: it tries grid values one parent at a time,
  as it does today.
- **The bell sits on the log-odds, not on the probability.** The heights are on the log-odds
  scale, which covers the whole number line, and σ turns any log-odds into a probability
  between 0 and 1. A bell over the heights therefore never puts probability outside 0–1.
  Example: log-odds 4 ± 1 means a probability between 0.95 and 0.993.
- **A category that is (almost) always drawn.** The data then keep pushing its log-odds towards
  +∞, so the hill has a slope but no top. Only the tiny ridge stops the climb: 0.001 is a belief
  that heights lie within about ±30 log-odds. The top lands far out and the hill is lopsided, so
  the bell describes it badly. Three rules handle this:
  1. **A weak belief that log-odds rarely go beyond ±10.** The ridge for categorical children is
     1/5² = 0.04, a spread of 5 log-odds, instead of 0.001 (D9). `BayesianFitting` already takes
     the ridge as a setting. `OutcomeNode` keeps 0.001, so its evidence does not change.
  2. **Levels that never occur are dropped,** because nothing can be learnt about them (their
     cut-point would run off to infinity). A child with only one level in the data is skipped,
     since every parent set would score the same.
  3. **A test on a child that is 98% one category** compares the Laplace evidence with a sampled
     one. This measures how much the remaining lopsidedness costs.

**Tests** (`categorical_nodes.feature`):
- `OutcomeNode` gives the same evidence after the move (the existing tests).
- A binary child: the logistic evidence prefers the true parent over a wrong one.
- Ordered logit and softmax: the Laplace evidence is close to a direct, sampled computation on
  a tiny example, as `test_evidence` does for `OutcomeNode`.
- Renumbering a nominal child's categories leaves its evidence unchanged.
- A binary child that its parent predicts perfectly (like the censoring flag) gives a finite
  fit.
- A child that is 98% one category: the Laplace evidence is close to the sampled one.
- A level that never occurs is dropped, and a child with a single level is skipped.
- An unseen category get's the average of existing ones

**Smoke:** F1 overall, and for arrows into categorical variables.

### Phase 3: categorical parents (P3)

**What:**
- **`Levels`** (new `terms/levels.py`): one height per level.
  - `ordered=True` penalises differences between neighbouring levels, λ × Σ (hₖ₊₁ − hₖ)²
    (ordinal smoothing, Gertheiss & Tutz 2009).
  - `ordered=False` penalises each height's distance from the average height,
    λ × Σ (hₖ − average)², since there is no order.
  - **What the penalty prefers.** Both versions pull the heights together.
    - **A very strong pull** ends with every level at the same height, the average, so the
      parent has no effect. That is the same as leaving the arrow out. Today's spline ends
      somewhere else: at a straight line, because second differences leave straight lines
      free (D10).
    - **λ is chosen by the evidence,** like a curve's, so the pull is strong only when the
      data show no real differences between levels.
    - **In between,** a level with few rows borrows from the others. Credit band 4 with 3
      borrowers is pulled towards band 3 (ordered) or towards the average (nominal), while a
      band with 300 borrowers barely moves.
  - It offers the same methods as `Spline` (weights, bending, flattening,
    zero_average_basis, band, outside, call), so `Node.store`, `ChangeFit` and `plot.py`
    keep working.
  - A level never seen in training predicts the average.
- **Choosing the term:** `Node` gets the kinds of its parents. `place_knots` builds a `Spline`
  for a continuous parent and a `Levels` term for a binary, ordinal or nominal one.
- **Surfaces:** `PairScreening` skips pairs that involve a categorical parent, for now (D6).

**Tests** (`levels.feature`):
- Renumbering a nominal parent's categories leaves the child's evidence unchanged.
- An ordinal parent with 5 levels recovers a step pattern.
- A binary parent gives a single difference between its two heights.
- An unseen level predicts the average.
- Re-ordering ordered categories decreases evidence

**Smoke:** F1, and for arrows out of categorical variables.

### Phase 4: a learnt noise spread and shape (P4)

**Now and after.** A node's score has two parts: how well it fits, and what its flexibility
costs. Phase 4 changes only the first part.

| | Now | After |
|---|---|---|
| The curves | fitted by least squares, λ chosen by the evidence | the same, unchanged |
| The fit term | −N/2 · log σ̂²: looks only at how spread out the leftovers are | Σ log f(leftoverᵢ): looks at their spread and their shape |
| The noise spread | one σ̂² for everyone in the node | a spread curve per parent: each person's spread is read off the curves at their parents' values |
| The noise shape f | always a bell | learnt from the leftovers divided by their spread; a bell is one possible result |
| Flexibility cost | the curves' | the curves', plus the cost of f's own flexibility |
| Roots | scored as a bell around their average | scored with their own learnt f |

**A toy: leftovers spread evenly between −1 and 1.**
- **Now:** a bell with the same spread gives each leftover an average log-density of −0.87.
- **After:** the learnt f is flat at height ½, so each leftover gets log ½ = −0.69.
- **Result:** with 1,000 rows the fit term rises by about 180 points, minus a few for f's
  flexibility, because the shape is now described correctly.

**Why the true direction scores better: two coins.** Flip two fair coins, x and n (0 or 1
each), and let y = x + n. So x → y, and the second coin n is the noise. Flexibility costs are
left aside, since they are about the same in both directions.

- **True direction, x → y.** A pair's score multiplies the probability of the root by the
  probability of the leftover. The root x is 0 or 1, ½ each. The leftover is n minus its
  average, −0.5 or +0.5, ½ each. Every row gets ½ × ½ = ¼.
- **Reverse direction, y → x.** The root is now y: 0 (¼), 1 (½) or 2 (¼). The curve predicts x
  from y as 0, 0.5 and 1. So the leftover is 0 when y is 0 or 2, and −0.5 or +0.5 when y is 1.
  Pooled, f says: leftover 0 half the time, −0.5 and +0.5 a quarter each.

  The same four rows, scored both ways:

  | Row: x, n | y | True: leftover y − (x + 0.5) | True: P(x) × f(leftover) | Reverse: leftover x − guess(y) | Reverse: P(y) × f(leftover) |
  |---|---|---|---|---|---|
  | 0, 0 | 0 | −0.5 | ½ × ½ = ¼ | 0 | ¼ × ½ = ⅛ |
  | 1, 1 | 2 | +0.5 | ½ × ½ = ¼ | 0 | ¼ × ½ = ⅛ |
  | 0, 1 | 1 | +0.5 | ½ × ½ = ¼ | −0.5 | ½ × ¼ = ⅛ |
  | 1, 0 | 1 | −0.5 | ½ × ½ = ¼ | +0.5 | ½ × ¼ = ⅛ |

  Every row gets ⅛, half of what the true direction gives it. With 1,000 rows that is
  1,000 × log 2 ≈ 690 points in favour of x → y.
- **Where the probability goes.** Multiplying P(y) by f(leftover) assumes the two are
  unrelated, so the reverse model also gives probability to pairs that never happen: y = 0
  with leftover ±0.5, y = 1 with leftover 0, and y = 2 with leftover ±0.5. Those pairs take
  half of its probability. In the true direction x and n really are unrelated, so all four
  pairs happen and nothing is wasted.
- **Why today's score cannot see this.** The bell-shaped score looks only at variances.
  - True direction: var(x) × var(leftover) = ¼ × ¼ = 1/16.
  - Reverse direction: var(y) × var(leftover) = ½ × ⅛ = 1/16.

  It is a tie, and for a straight line it always is. This is the gap between F1 0.24 (no
  background) and 0.48 (roots given).
- **The one case that really ties.** If x and n are both bell-shaped, the reverse leftover is
  also unrelated to y (only bells have this property), so neither direction wastes
  probability. The data then cannot tell the direction, and the doubtful flag is the right
  answer.
- **Continuous variables work the same way,** with densities instead of probabilities. For
  y = 2x + noise, with x and the noise spread evenly, the reverse leftover's range shrinks at
  the ends of y's range, and the same waste appears (Kpotufe et al. 2014, the same idea as
  LiNGAM).

**How it is built:**
- **`NoiseDensity`** (new `nodes/noise_density.py`) learns f, the shape of a set of numbers.
  - **What kind of function f is.** A density: a smooth, positive curve over the leftover
    values, with total area 1. Its height says how common each leftover is. It is built as
    f(r) = e^s(r) / Z:
    - s is a P-spline with about 20 bumps, the same kind of curve as an edge;
    - e^ keeps f positive;
    - Z is the number that makes the area 1.
  - **How it is learnt:**
    1. Sort the N leftovers into about 100 equal bins and count each bin.
    2. The model says bin j should hold about N × bin width × f(centre of bin j) leftovers.
       The counts are fitted with Poisson regression, the standard model for counts (Eilers &
       Marx 1996, density smoothing).
    3. Newton steps find the heights of s. The Poisson log-likelihood is concave in them, so
       there is one top.
    4. The penalty is on third differences of the heights, and λ is chosen by the evidence
       (Laplace), as for the other nodes.
  - **Why third differences.** Third differences are zero for any parabola, so a parabola
    costs nothing, and e^parabola is exactly a bell. A very strong penalty therefore gives
    today's bell, and the data decide how far to move away from it: a flat top for even
    leftovers, a long tail for skewed ones, a sharp peak with heavy tails for outliers.
- **`FlexibleNode(Node)`**:
  1. Fits the curves exactly as now, with λ chosen with the bell-shaped evidence.
  2. Fits `NoiseDensity` once to the final residuals.
  3. Replaces the bell-shaped term in the evidence with the sum of log-densities of the
     residuals. The curves' own Occam terms (½ log|P| − ½ log|A|) are kept, and the
     density's Occam term is added.

  Roots get the same treatment, with residual = x − constant. That is where the bell
  assumption hurts most.
- **Bin width:** at least the noise floor's scale (Phase 1), and at least the child's rounding
  step when its values lie on a grid. Spikes from rounding or piled-up values then earn
  nothing.
- **Where it is used:** `EvidenceScore` uses `FlexibleNode` for continuous variables when the
  benchmark asks for it. `ChangeFit` (drift) keeps the bell-shaped evidence.

**Tests** (`noise_density.feature`):
- Bell-shaped numbers: the learnt density stays a bell, with λ at the top of the grid and a
  log-likelihood within about 1 of the bell's.
- Uniform numbers: the learnt density is flat, and its log-likelihood beats the bell's.
- A uniform cause with y = x² plus small noise: the total evidence prefers x → y with learnt
  noise. The test also records what the bell-shaped score says.
- The evidence approximation is close to a direct computation on a toy.

**Smoke:** the gap between "no background" and "roots given" should shrink. That gap
measures this problem.

#### The spread curve (added 2026-11-24)

**Why.** The first smoke test of the learnt shape lowered F1 in 7 of 8 runs. The shape works
when its assumption holds (all 70 synthetic pairs with noise added after the formula), but
CausalArena's noise changes its spread with the parents: binomial counts, zero-inflated
counts, multipliers. Then the leftovers are linked to the cause in the true direction too,
and the learnt shape is confidently wrong: 39% right when the spread grows with the cause,
0% with a sigmoid-shaped spread. On the true continuous edges of the two smoke files it
pointed 21–24% of the directions right, against 32–39% for the bell.

**What.** Each continuous node gets a second set of curves, for how far its values stray
from the prediction:

    x = constant + Σ f_parent(x_parent) + spread(parents) × noise
    log spread² = c + Σ g_parent(x_parent)

This is the location-scale noise model (Immer et al. 2023). It keeps the noise independent of
the parents, it is identifiable, and it is invertible (noise = (x − prediction) / spread), so
abduction for recourse still works.

- **`NoiseSpread`** (new `nodes/noise_spread.py`): one spread curve per parent (`Levels` for a
  categorical parent), fitted to the squared leftovers: Newton on Σ [−½ log v − r²/(2v)] minus
  the penalty, with v = spread² and r the leftover. The penalty is on first differences
  (flattening), so a strong pull gives a flat curve: one spread for everyone, today's model.
  λ per curve is chosen by the Laplace evidence, as for every other curve.
- **`FlexibleNode`**, step by step:
  1. The mean curves, as now (least squares, λ by the bell-shaped evidence). Leftovers r.
  2. The spread curves, fitted to r². The noise floor is added to each v.
  3. The leftovers divided by their spread, z = r / √v, should look alike for everyone.
     `NoiseDensity` learns their shape f.
  4. evidence = Σ [log f(zᵢ) − ½ log vᵢ] − the mean curves' prior cost + their Occam term
     + the spread curves' Occam term + f's Occam term. The −½ log v is there because
     dividing by √v squeezes the axis.

  Roots get a spread with no curve: one constant, as before.
- **Nested.** Both new parts are pulled towards the simple model: the spread towards flat,
  the shape towards a bell. The evidence decides, per variable, how far to move. Choosing
  between "spread + bell" and "spread + learnt shape" by a benchmark score is avoided on
  purpose.
- **Bin width for z:** never narrower than two scales, so an exact or rounded formula still
  earns no spike:
  - the floor's: share × √(2πe), with share = c / (c + mean v), the part of each leftover's
    variance that the floor adds. While the floor is negligible the bins stay fine; when it
    dominates, the narrowest shape allowed is a box as uncertain as a bell of spread 1
    (√(2πe) wide), which gives back the bell floor's cap.
  - the child's rounding step, divided by the typical spread.

  Each leftover is scored by the probability of its bin divided by the bin's width (what the
  Poisson fit estimates), and wide bins are centred on the values. Reading f off at the exact
  value was wrong for values on a grid: they fell between two bins, in the troughs of a smooth
  curve over spikes. `closing_cost_subsidy` (11 values) scored 650 below the bell it contains;
  after the fix it scores 130 above it.

  Wider bins cover only the range of z (at least 10 bins). The first version kept 100 bins
  whatever their width, so a forced width stretched the grid far past the data and the
  bumps became too coarse for any shape but a bell. That was a second reason the first
  smoke test went wrong: with the old rule, the learnt shape without a spread curve got 39%
  of the growing-spread pairs right, and 83% without the rule.
- **Simple version first:** the spread is fitted once, after the mean curves. The joint fit
  (refit the mean with weights 1/v, then the spread, until they settle) is more exact and
  left for later.

**What it does not fix:** distortions after the noise (a logistic link, rounding),
zero-inflation, and interactions between the causes and the noise.

**Tests** (`noise_spread.feature`):
- Noise whose spread grows with the cause: the spread curve follows it.
- Noise with a constant spread: the spread curve stays flat.
- A sigmoid-shaped spread: the bell-shaped score prefers the wrong direction by more than
  100, and the learnt spread and shape prefer the right one.
- A rounded formula's gain stays within 10% of the bell floor's cap.
- Before the smoke test, on 54 synthetic pairs (three formulas, three spreads, two noise
  shapes, three seeds): the learnt spread and shape point the right way in all of them; the
  bell in 67% (constant spread), 56% (growing) and 33% (sigmoid).

**Smoke:** F1 against the first Phase 4 smoke and against Phase 3.

### Phase 5: interventional rows (P5)

**What:**
- `GraphSearch` and `EvidenceScore` accept `intervened`: for each variable, a yes/no per row.
  Each child is fitted only on the rows where it was not set.
- A variable that was set does not depend on its parents in those rows, so they tell nothing
  about its equation. This is the interventional score of GIES (Hauser & Bühlmann 2012).
- No row is wasted: a row that sets the cause still counts for the effect's equation.
- The benchmark passes `data.intervened` for obs_int files. CD-NOD already uses the same flags
  through C.
- **Not a leak:**
  - The flags come from the data file itself. Each value in `x` carries a second channel that
    says whether an intervention set it (`ArenaData.load`). They are not read from the true
    graph `g`.
  - A flag says which variable was set in a row, never what that variable causes.
  - Knowing the targets is the standard setting of interventional discovery (GIES).
  - CD-NOD in this benchmark already receives the same flags. PC pools the rows without them,
    and the report should say so.

**Tests:** a straight-line pair x → y with bell-shaped noise. Without interventions the arrow
is doubtful (the data cannot tell). With 200 rows that set x, x → y wins by more than the
margin.

**Smoke:** obs_int F1 against obs_only F1.


### Phase 6: a search over orders (added 2026-11-24)

**Why.** Kairos' search adds the best arrow, one at a time, and never takes one back. An early
arrow in the wrong direction can then block true arrows, because they would close a loop. A
check on the two smoke files showed that this is not yet what limits kairos there: the graph
greedy found scored higher than the true graph under the same score, in all four cases (both
files, bell and learnt), so the score is what prefers wrong graphs. On synthetic data, where
the score is right, greedy does leave a lot on the table.

**What.** `OrderSearch` (Teyssier & Koller 2005) searches over orders of the variables instead
of over arrows:
- An order says who may be a parent of whom. Each variable takes its best parent set among the
  variables before it, so every order gives a graph without loops.
- **`ParentSets`** gives each variable's best parents among those allowed. Parents come from its
  10 strongest single arrows (CAM's preliminary neighbourhood selection). Every set of up to 2 of
  them, and greedy's own parents, are scored once; larger sets are grown one parent at a time
  among the allowed candidates and then trimmed, with no cap on their size. All fits are cached.
  A first version capped sets at 4 parents; greedy's graphs on the smoke files have variables
  with 5–8 parents, so every swap looked worse and the search never moved.
- The search swaps two neighbours in the order; a swap changes only their two parent sets, so it
  costs two look-ups. It takes the best swap that is not tabu, even when it lowers the score,
  stops after 20 steps without a new best, and restarts 5 times from the best order shaken by
  10 random swaps.
- It starts from greedy's graph (its order and its parent sets), so it never scores below
  greedy. The pruning and the doubtful flag then run as before.
- `GraphSearch(order_search=OrderSearch())`; the benchmark uses it with `KairosSearch(search="order")`.

**Other options from the literature, for later:** exact search (dynamic programming, integer
programming as in GOBNILP) when parent sets are bounded; permutation search such as BOSS
(Andrews et al. 2023); order or partition MCMC (Friedman & Koller 2003; Kuipers & Moffa 2017),
which averages over graphs with the evidence and would turn "doubtful" into a probability per
arrow; order-first methods built for these noise models (CAM; SCORE and NoGAM; HOST and
SkewScore for noise whose spread changes with the parents).

**Tests** (`order_search.feature`):
- The collider trap: greedy builds a chain for x → z ← y; the order search finds the collider
  with a higher total evidence.
- A random 8-variable graph with curved arrows and non-bell noise: greedy gets part of it wrong,
  the order search recovers it exactly and never scores below greedy. On six such graphs greedy's
  F1 was 0.50–1.00, the order search's 0.95–1.00, and its total equalled the true graph's.
- The best parent set within an allowed set, and a swap's computed change, against direct
  computation.

**Smoke:** F1 against greedy, same scoring.

## 9. Files

| Phase | New | Changed |
|---|---|---|
| 0 | none | `discovery/background.py`, `discovery/graph_search.py`, `benchmarks/causal_arena/methods/kairos.py`, `tests/features/causal_arena.feature`, `tests/test_causal_arena.py` |
| 1 | `nodes/noise_floor.py`, `tests/features/noise_floor.feature`, `tests/test_noise_floor.py` | `nodes/node.py`, `nodes/fitting.py`, `nodes/flexible.py`, `nodes/__init__.py`, `terms/spline.py`, `terms/surface.py`, `discovery/evidence_score.py`, `discovery/graph_search.py`, `methods/kairos.py` |
| 2 | `nodes/laplace.py`, `nodes/binary.py`, `nodes/ordinal.py`, `nodes/category.py`, `tests/features/categorical_nodes.feature`, `tests/test_categorical_nodes.py` | `nodes/outcome.py`, `nodes/__init__.py`, `discovery/evidence_score.py`, `discovery/discovered_graph.py`, `discovery/background.py`, `methods/kairos.py` |
| 3 | `terms/levels.py`, `tests/features/levels.feature`, `tests/test_levels.py` | `nodes/node.py`, `nodes/screening.py`, `terms/__init__.py` |
| 4 | `nodes/noise_density.py`, `nodes/noise_spread.py`, `nodes/flexible.py`, `tests/features/noise_density.feature`, `tests/test_noise_density.py`, `tests/features/noise_spread.feature`, `tests/test_noise_spread.py` | `nodes/__init__.py`, `discovery/evidence_score.py`, `methods/kairos.py` |
| 6 | `discovery/parent_sets.py`, `discovery/order_search.py`, `tests/features/order_search.feature`, `tests/test_order_search.py` | `discovery/graph_search.py`, `discovery/__init__.py`, `methods/kairos.py` |
| 5 | `tests/features/interventional_rows.feature`, `tests/test_interventional_rows.py` | `discovery/evidence_score.py`, `discovery/graph_search.py`, `methods/kairos.py` |

Every phase also updates the README's kairos section. Paths without a folder prefix are under
`src/causal_model/`; `methods/kairos.py` is under `src/benchmarks/causal_arena/`.
