# Issue 2: the input model is wrong for categories

## The problem

A spline assumes its input is a number on a line, where nearby values behave
alike. kairos uses one for every parent, and the benchmark passes every variable
as a plain number (`methods/kairos.py` ignores `data.kinds`). For categorical
parents this assumption is false or wasteful.

### 2a. Nominal parents: the order is made up

**Example.** `earliest_event_type_true` is coded 0 = none, 1 = refinance,
2 = sale, 3 = default. A spline treats "sale" as lying between "refinance" and
"default", and its smoothness penalty pulls neighbouring codes towards similar
effects. Renumber the codes (default = 1, refinance = 3) and the fit changes, and
so can the graph.

The right model is one height per category: a step function, not a curve.

### 2b. Ordinal parents: a curve over a handful of points

**Example.** `credit_score_band` takes 5 values. A spline with 8 intervals has data
at 5 points only; most of its bumps are set by the smoothness penalty alone. The
order is real, so this works, but it is wasteful, and the evidence's penalty
terms are computed as if the input were continuous.

### 2c. Binary parents: harmless

A spline over two values reduces to two heights, i.e. one effect. No problem.

## Summary

| Parent kind | Example | Problem |
|---|---|---|
| nominal | event type | wrong: arbitrary order |
| ordinal | credit band | wasteful |
| binary | application attempted | none |

## Possible remedies

- One height per category for nominal parents (with a penalty shrinking the
  heights towards each other, not towards their neighbours).
- Ordinal parents: keep the spline with fewer intervals, or one height per level
  with a penalty on differences between adjacent levels.
- Read the measurement scale from the data (`ScenarioData.kinds`) and pick the
  term type per parent.
