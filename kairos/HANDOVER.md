# Handover: kairos' causal discovery

Written 2026-11-25, after the core was simplified. Read this first, then `docs/causal_kan.md` (how it
works) and `README.md` (what each file does). `src/causal_model/TYPES_AND_NOISE_PLAN.md` is the older plan; its top says what still holds.

## 1. In short

- **kairos** is the thesis' causal KAN: one smooth curve per arrow (Bayesian P-splines), one
  equation per variable. The goal is fair recourse that stays valid when the world drifts.
- **The problem was:** the score ranked wrong graphs above the true one. **The cause:** kairos could
  write only sums of curves with bell noise. When a true equation is not one (z = e^(x + y), a gate,
  noise that is not a bell), it fits badly, while a backwards graph may fit well.
- **The fix:** let an equation take the forms true equations have (a scale, surfaces, a noise size and
  shape), and replace the greedy search by one that can also remove, reverse and restart.
- **Where it stands:** made-up worlds F1 0.50–0.62 → 0.86–1.00; smoke files 0.44 → 0.61 on average.
  The score is right or close on made-up worlds and nearly so on healthcare (110 points), but **on
  mortgage a wrong graph still scores 1,589 points above the true one** (section 6).
- **State:** 142 tests pass, none expected to fail, pyright 0 errors. **Nothing is committed.**
  `src/causal_model` went from about 2,350 lines to 1,938.

## 2. Rules for working on this

- Do not commit. Do not push.
- Never delete files unless told to. When told, move them to `~/.Trash`.
- Ask before any download or install. About 1 MB of CausalArena (the manifest, the mortgage
  scenario and one healthcare scenario) was approved and is already downloaded (section 9).
- **Smoke tests only.** Full benchmark runs are Tiago's, on a remote machine.
- Do not save previous versions of files before editing.
- Tiago edits the same files in VS Code: list every changed file so they can use Compare.
- Keep Tiago's own text verbatim.
- Explain from scratch: define every concept, with a toy example. Short answers are valuable.
- Code: small classes with one job each (the python-development skill). Tests are behaviour
  scenarios (`tests/features/*.feature`, steps in `tests/test_*.py`).
- **No noise model tuned to the benchmark.** Choices come from reasoning or synthetic data;
  CausalArena only evaluates them.

## 3. How to run (from `kairos/`)

    venv/bin/python -m pytest -q                                              # about 25 s
    venv/bin/pyright --pythonpath venv/bin/python src tests scripts
    venv/bin/python -I diagnostics/smoke/smoke.py data/causal_arena           # 8 runs, about 7 min
    venv/bin/python -I diagnostics/smoke/gap.py mortgage                      # is the true graph ahead?
    venv/bin/python -I diagnostics/worlds/run.py 700 730 full sums-only       # made-up worlds
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run --help    # real runs download data
    bash scripts/run_causal_arena.sh                                          # the full run, in the background

The full run uses 5 processes (`WORKERS=4` for 4), keeps going after logout, and goes on from the
finished files when started again with the same `OUT` and methods. It replaced the tmux and merge scripts
(now in `diagnostics/set_aside/scripts/`). Estimated from two files on this Mac: 21 hours of work for
everything, about 4 hours with 5 processes; about 3 hours for kairos alone. Never run on Linux yet.

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` for anything parallel. The
methods in `run.py` are `kairos`, `pc-fisherz`, `pc-rcit`, `cdnod-fisherz`, `cdnod-rcit` and
`cdnod-authors` (hours per file).

## 4. The core

1. **A variable's equation.** scale(x) = constant + curves + at most two surfaces + noise.
   - **Scale:** x, log x or the log-odds of x, whichever the evidence prefers (`Scale`). Scores on
     different scales are comparable because of the Jacobian (`Scale.stretch`).
   - **Surfaces:** for what depends on two parents together, categorical ones included (a gate is one
     curve per level). `PairScreening` adds one only if the evidence rises by more than 3.
   - **Smoothness:** one λ per curve and per surface, set to noise variance × freedom / bending until
     the evidence stops rising (no grid any more).
   - **Noise:** fitted as a bell, then its size (may grow with the prediction) and its shape (15 fixed
     ones) are read off the leftovers (`NoiseModel`).
   - **Floor:** x and each term are trusted only to 1% of their spread, so an exact formula cannot
     score without limit, and scores the same whichever way round it is written.
   - Yes/no, ordered and unordered variables keep their own likelihoods (logistic, ordered logit,
     softmax).
2. **Score of a graph** = sum of the variables' evidence − 4 per arrow (`GraphSearch.price`).
3. **Search** (`GraphSearch`): from no arrows, the best single change (add, remove, reverse) until
   none helps; then 20 restarts from the best graph with 6 random arrows removed or reversed.

## 5. What was decided, and the evidence

Everything was chosen on made-up worlds (`diagnostics/worlds/`), never on CausalArena. Final check on
30 fresh worlds of each kind (seeds 700–729; F1, standard error about 0.02):

| | general | exact formulas | nearly straight arrows | lopsided noise | changing noise size |
|---|---|---|---|---|---|
| **Core** | **0.935** | **0.860** | **1.000** | **0.978** | **0.911** |
| without scales | 0.878 | 0.832 | 1.000 | 0.978 | 0.876 |
| without surfaces | 0.803 | 0.839 | 1.000 | 0.974 | 0.859 |
| bell noise of one size | 0.891 | 0.816 | 0.734 | 0.577 | 0.866 |
| without the floor | 0.934 | 0.834 | 1.000 | 0.978 | 0.911 |
| sums of curves, bell noise | 0.678 | 0.581 | 0.821 | 0.705 | 0.701 |
| HSIC order search (KaCGM-like) | 0.386 | 0.507 | 0.913 | 0.968 | 0.348 |

The kairos of 24 November, on other seeds of the first two kinds: 0.53 (bell) and 0.62 (learnt noise)
on general worlds, 0.50 and 0.77 on formulas.

**Removed** (code in `diagnostics/set_aside/`, with its tests), because it made no difference or
something simpler did better:

| Removed | Instead | Why |
|---|---|---|
| carried noise floor, slopes | each term trusted to 1% of its spread | same F1; no slopes needed |
| learnt density and spread curves | 15 shapes, one growth number | as good or better, and 2 to 3 times faster |
| λ grid, plain fitting | fixed point | never worse, up to 258 points better (the grid stopped at 0.1) |
| greedy + pruning | climb with a price per arrow | F1 0.89 → 0.93 on fresh worlds (pruning against a price) |
| order search, parent sets | restarts | 0.80 → 0.84 |
| three λ per surface, saddle | one | nothing lost |
| OutcomeNode, log-odds option, `kairos-order` | BinaryNode, the fitting's ridge | duplicates |

Tried and not kept: a separate equation per level of a gate, parents on their own scale or on ranks,
surfaces for categorical children, adding random noise to the parents instead of the floor (it broke
ties at random).

## 6. Results on the smoke files

F1, one sample per file, so differences of 0.05 mean little.

| File | 24 Nov, bell | 24 Nov, learnt | Now |
|---|---|---|---|
| Mortgage, observational, no background | 0.27 | 0.26 | 0.45 |
| Mortgage, observational, roots given | 0.48 | 0.51 | 0.70 |
| Mortgage, interventions, no background | 0.26 | 0.17 | 0.48 |
| Mortgage, interventions, roots given | 0.45 | 0.40 | 0.61 |
| Healthcare, observational, no background | 0.38 | 0.42 | 0.62 |
| Healthcare, observational, roots given | 0.52 | 0.51 | 0.69 |
| Healthcare, interventions, no background | 0.46 | 0.41 | 0.54 |
| Healthcare, interventions, roots given | 0.69 | 0.52 | 0.76 |
| Average | 0.44 | 0.40 | **0.61** |
| Seconds for all eight | 408 | 775 | 418 |

Others, same files: sums of curves with bell noise and the new search 0.51; PC (Fisher z, 4 runs
without background) 0.38, where kairos has 0.52; HSIC order search 0.34.

**The noise model does not help on these two files** (0.61 against 0.63 with bell noise). It is kept
because it is what finds the direction when noise is not a bell (table above).

**Is the true graph ahead?** (observational, no background; worth found − worth of the truth, `gap.py`)

| | 24 Nov, bell | Now | Climbing from the truth ends at F1 |
|---|---|---|---|
| Mortgage | 3,262 | 1,589 | 0.71 |
| Healthcare | 54 | 110 | 0.75 |

The two columns use different scores, so only their signs compare. Only 30 of 44 and 24 of 40 true
arrows earn their price, so F1 cannot reach 1 on these files.

## 7. Open problems and next steps

1. **Three parents at once** (a gate on a product). On mortgage, `realized_refinance_savings_npv`
   prefers wrong parents by 985 points: refinanced × rate × balance × term is more than two surfaces
   over pairs can write. A separate equation per level of a gate was tried and lowered F1 elsewhere.
2. **Balance and LTV bucket** (mortgage): the reversed pair is about 180 points ahead. Cause not found.
3. **`monthly_refinance_hazard`** prefers an extra parent (event type, 439 points). Cause not found.
4. **Surfaces cost time**: about half of the search.
5. **Recourse:** `Node.noise` is the leftover on the scale; a counterfactual for a node whose noise
   grows must also divide by its size. Counterfactuals are not built yet.
6. **Drift:** `ChangeFit` uses the bell evidence and does not choose scales.

## 8. KaCGM

KaCGM (Almodóvar et al. 2026, arXiv 2603.20184, github.com/aalmodovares/kacgm) does **not** discover
graphs: it assumes the causal graph is known and correct, and checks it afterwards by testing (HSIC)
whether the recovered noise is independent of the parents. Its code needs torch and pykan, which are not
installed, so it was **not run**. `diagnostics/kacgm/` holds two stand-ins built from its ingredients
with kairos' splines: its node (one curve per parent, additive noise; logistic, ordered or softmax for
discrete variables) as the score in the same search (`sums-only`), and its HSIC check as a search for
the causal order (`resit.py`, as in RESIT, Peters et al. 2014). Results: sections 5 and 6.

## 9. Diagnostics

`diagnostics/` is not part of kairos and has no tests.

| In `diagnostics/` | What |
|---|---|
| `worlds/` | the made-up worlds and the ablations (`run.py`, `results.txt`) |
| `smoke/smoke.py`, `smoke/gap.py`, `smoke/core.txt`, `smoke/gap_*.txt` | the smoke test and the score gap, now |
| `kacgm/` | the KaCGM comparison |
| `set_aside/` | code and tests taken out of the core on 25 November |
| `why/`, `checks/`, the other `smoke/` files | before 25 November: they import removed code and no longer run |

## 10. Files changed on 25 November

- **New:** `docs/causal_kan.md`, `scripts/run_causal_arena.sh`, `parallel.py`, `nodes/scale.py`, `nodes/noise_shape.py`, `nodes/noise_model.py`; `tests/` `scale`,
  `right_graph`, `fitting`, `noise_shape` (`.feature` and `test_*.py`)
- **Rewritten:** `nodes/fitting.py`, `nodes/screening.py`, `nodes/node.py`, `terms/surface.py`,
  `discovery/` `node_maker.py`, `evidence_score.py`, `graph_search.py`, `discovered_graph.py`,
  `methods/kairos.py`, `src/simulator.py`
- **Edited:** `nodes/` `laplace.py`, `binary.py`, `ordinal.py`, `category.py`, `__init__.py`;
  `terms/spline.py`; `discovery/` `background.py`, `graph.py`, `__init__.py`; `drift/change.py`; `plot.py`;
  `simulation/drift.py`; `methods/kairos_inputs.py` (raw values instead of standardised); `run.py`;
  the tests of drift, evidence, screening, categorical nodes, discovery, interventional rows,
  CausalArena; `tests/helpers.py`; `README.md`; the plan (a note at its top); this file
- **Moved to `diagnostics/set_aside/`:** `nodes/` `noise_floor.py`, `noise_density.py`,
  `noise_spread.py`, `flexible.py`, `outcome.py`; `discovery/` `order_search.py`, `parent_sets.py`;
  the tests of noise floor, noise density, noise spread and order search; the two old `scripts/`
- **Tests whose expectation changed:** world 6 without roots no longer fails; the pruning scenario
  became a climbing one; "a node without a smoothness belief is refused" went with plain fitting;
  kairos now gets raw values.

(Paths without a folder are under `src/causal_model/` or `src/benchmarks/causal_arena/`.)
