# How the causal KAN works

Every number below comes from running kairos on 10,000 people from the small world in
`src/simulator.py`, unless the text says otherwise.

## 1. The idea

A causal KAN is a set of equations, one per variable, that say how the world produces each
value. The small world:

    income  = anything between 0 and 4
    savings = 0.5 · income² + noise
    Y       = 1 (loan paid back) with a chance that rises with income and savings

As a graph, where an arrow means "is a direct cause of":

    income ──────────► Y
      │                ▲
      └──► savings ────┘

Each arrow carries **one curve**: how the cause pushes the effect. The pushes are added at
the effect. So the whole model can be looked at: one small plot per arrow.

| Word | Meaning | Example |
|---|---|---|
| Variable | something measured for each person | income |
| Parent, child | the two ends of an arrow | income is a parent of savings |
| Root | a variable with no parents | income |
| Push | how much a parent moves the child, for one person | income 3 pushes savings up by 1.83 |
| Noise | what the parents do not explain | this person saves 0.49 more than expected |

## 2. The parts

    people (a table of data)
        │
        ▼
    discovery/    finds which arrows exist             → a graph
        │             scores parent sets with ▼
        ▼
    nodes/        one equation per variable            → curves, surfaces, scale, noise
        │             built from ▼
        │         terms/  curves, levels, surfaces
        ▼
    kan.py        the nodes in causal order: the causal KAN
        │
        ├──► plot.py    one plot per arrow
        └──► drift/     did an equation change between two periods?

Folders are under `src/causal_model/`.

## 3. One variable's equation (`nodes/node.py`)

    scale(x) = constant + Σ curves + Σ surfaces + noise

Savings, as fitted, for a person with income 3 and savings 5:

    savings = 2.68 + curve(income) + noise
       5    = 2.68 +     1.83      + 0.49

The truth is 0.5 · 3² = 4.5, and the model predicts 4.51. The noise, 0.49, is what makes this
person different from others with the same income. It matters for recourse: "what if *you* earned
more?" keeps your noise.

### 3.1 A curve (`terms/spline.py`)

A curve is a sum of small bumps. Each bump sits at a fixed place on the axis and has a **height**,
which is a number the data choose:

    curve(x) = Σ weight_k(x) · height_k

At any x at most four bumps count. With bumps at −1, 0, 1, 2, 3, 4 and 5, income 2.4 gives:

| Bump at | −1 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| Weight | 0 | 0 | 0.036 | 0.539 | 0.415 | 0.011 | 0 |

So learning a curve means choosing a few heights, and the equation is a plain weighted sum of them.
Each curve averages zero over the people, and the constant holds the level: "1.83" reads as "1.83
above the average person". Beyond the data the curve goes on as a straight line.

### 3.2 A categorical parent (`terms/levels.py`)

Gender or an event type has no "in between", so it gets **one height per level** instead of a curve.
Ordered levels (credit bands) are pulled towards their neighbours, unordered ones towards the average.

### 3.3 A surface (`terms/surface.py`)

Some effects depend on two parents together. A gate is the clearest case: w = flag · 2x moves with x
only if the flag is on. Curves are added, so they cannot say "only if". A surface stores a height for
each **pair** of values:

| | Share of w left unexplained |
|---|---|
| Two curves | 61% |
| Two curves and one surface | 2% |

(Made-up data, 1,000 people.) A surface holds only what depends on both parents; the rest stays in the
curves, so each effect has one place to be read. `nodes/screening.py` decides which pairs get one, at
most two per variable: it looks for groups of people (high x and flag on, ...) whose leftovers are all
too high or all too low, and keeps a surface only if the score rises by more than 3 (section 4).

### 3.4 The scale (`nodes/scale.py`)

Some equations are a sum only after undoing a bend. z = e^(x + y) is not a sum of curves, but
log z = x + y is. Kairos tries x, log x (positive values) and log-odds (values between 0 and 1, like
probabilities), and keeps the one with the best score:

| z = e^(x + y), made-up | Score |
|---|---|
| plain | 2,127 |
| **log** | **2,310** |

Taking the log squeezes large values together, which makes every error look smaller. The score
subtracts exactly that squeeze (the Jacobian), so the comparison is fair.

### 3.5 The noise (`nodes/noise_model.py`, `nodes/noise_shape.py`)

The curves are fitted as if the noise were a bell of one size. Then the noise is read off the leftovers:

- **Size:** it may grow or shrink with the prediction (large incomes vary more). One number.
- **Shape:** one of 15: peaked, bell or flat; leaning right, left or neither.

Why it matters: with bell noise and a straight line, "x causes y" and "y causes x" fit equally well.
With any other noise, only the true direction leaves noise that looks the same for everyone.

### 3.6 Yes/no, ordered and category variables

Their value is not a number plus noise, so they have their own equations:

| Variable | File | Equation |
|---|---|---|
| yes/no (and the outcome Y) | `nodes/binary.py` | chance of 1 = σ(sum) |
| ordered (credit band) | `nodes/ordinal.py` | the sum pushes a person up the levels, cut-points mark where each starts |
| category (event type) | `nodes/category.py` | one sum per category; the largest is the most likely |

σ(s) = 1 / (1 + e^−s) turns any number into a chance between 0 and 1. For the person above:

    sum = −0.65 + 1.02 (income) + 1.17 (savings) = 1.54
    σ(1.54) = 0.823         (the true chance is 0.818)

## 4. Choosing the heights, and the score (`nodes/fitting.py`)

**Heights.** They are chosen to leave small leftovers while the curve stays smooth. How smooth is a
strength **λ** per curve: λ near 0 trusts only the data (wiggly), a huge λ allows only straight lines.
Savings' curve got λ = 11.

**The score.** The *evidence* of an equation is how probable the data is under it, as a logarithm.
Adding 4 to the evidence means the data is e⁴ ≈ 55 times more probable. It rewards small leftovers and
charges for freedom, so a wiggly curve that only fits the noise loses:

    evidence = −N/2 · log(size of the leftovers²)   −   the price of the freedom used
               + what the noise's size and shape add   + the scale's correction

λ is the value with the highest evidence. Kairos finds it by repeating

    λ = noise² · freedom / bending

(freedom: how many directions of the curve the data decided; bending: how much the curve bends) until
the evidence stops rising. Nothing is set by hand, and the same score decides λ, surfaces, scale, noise
and arrows.

**The floor.** An exact formula leaves no leftovers, and the logarithm would go to infinity. So nothing
is trusted beyond 1% of its spread: the leftovers are never taken smaller than that.

**Bands.** The fit also says how unsure each height is, so every curve has a 95% band: income 3 pushes
savings by between 1.81 and 1.85.

## 5. Finding the graph (`discovery/`)

**Score of a graph** = the sum of its variables' evidence − 4 per arrow. For income and savings alone:

| | income | savings | Total |
|---|---|---|---|
| income → savings | −134.5 (no parents) | 12,035.5 (from income) | **11,901** |
| savings → income | 17,880.8 (from savings) | −7,182.6 (no parents) | 10,698 |

The true direction is ahead by 1,203. Why a direction can be known from data: the equations of the true
graph fit simply (one curve, noise of one kind). The reversed equation is of a stranger form, because
it has to undo a curve with noise inside, and it fits worse.

**The search** (`graph_search.py`):

1. Start with no arrows.
2. Make the single change (add, remove or reverse an arrow) that raises the score most. Repeat until
   none does. Loops are never allowed.
3. Shake the best graph (6 random arrows removed or reversed) and climb again, 20 times. Keep the best.
4. Flag an arrow as **doubtful** if its reverse scores within 3.

What it found:

    income  ← no parents
    savings ← income (gain 19,218)
    Y       ← income (18), savings (16)

The gain is how much evidence is lost without the arrow. With only 2,000 people it does not find
savings → Y: savings is almost a function of income, so few people show the difference.

**What is given, not found** (`background.py`): which variable is the outcome, which are roots (if
known), and each variable's kind (continuous, yes/no, ordered, category).

**Interventions:** in a row where a variable was set from outside, its parents did not produce it. That
row is left out of that variable's equation only.

| File | Job |
|---|---|
| `background.py` | what is known beforehand |
| `node_maker.py` | the right node for each kind of variable |
| `evidence_score.py` | the score of one variable with one parent set, remembered |
| `graph.py` | the arrows so far, and the loop check |
| `graph_search.py` | the search |
| `discovered_graph.py` | the result: `summary()`, and `build()` for the causal KAN |

## 6. Did the world change? (`drift/change.py`)

`ChangeFit` takes people from before and after and fits one equation to both, with an extra **change
part** per curve that counts only for the people after. A change is kept only if the data become more
than e³ ≈ 20 times more probable.

Example: 5,000 people before and 5,000 after savings changed from 0.5 · income² to 0.3 · income².

| | Income 2 | Income 4 |
|---|---|---|
| Before: predicted (true) | 1.99 (2.0) | 8.02 (8.0) |
| After: predicted (true) | 1.23 (1.2) | 4.81 (4.8) |

Flagged: the income curve (gain 6,362) and the level (6,999). A shift in *who the people are*, with no
equation changed, is not flagged (tested in `tests/features/drift.feature`). This is why each arrow has
its own curve: the model says **which** mechanism changed.

## 7. Use it

```python
from src.causal_model.discovery import Background, GraphSearch
from src.causal_model.plot import KANPlot

found = GraphSearch(Background(outcome="Y")).run(data)   # data: {name: array}, one value per person
print(found.summary())
model = found.build()                                    # the causal KAN
model.fit(data)
model.node("savings").predict({"income": array})         # what the parents explain
model.node("savings").noise(people)                      # what they do not
KANPlot(model).figure().savefig("curves.png")
```

## 8. What it cannot do (yet)

- **Three parents at once** (a gate on a product): surfaces take two.
- **Hidden causes:** it assumes every common cause is measured.
- **Directions the data do not show:** a straight line with bell noise has none. The flag says so.
- **Counterfactuals and recourse** ("what if this person's income changed?") are not built. The pieces
  are: the noise is recovered, and the equations are in causal order.
- **Online updates:** everything is fitted on a batch.
- **Fairness** is kept out on purpose: the model describes the world; which routes are unfair is decided
  in the recourse search and the monitor.

How well it finds graphs, and what is still wrong: `HANDOVER.md`, sections 5 to 7.
