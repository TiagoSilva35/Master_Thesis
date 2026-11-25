# Kairos: a causal KAN that notices drift

A causal graph whose equations are KAN curves: one curve per edge, summed at the
node, plus a surface for a pair of parents that interact.

    scale(x_j) = c_j + Σ φ_ij(x_i) + Σ ψ_abj(x_a, x_b) + u_j    (every continuous variable)
    P(Y = 1) = σ(c_Y + Σ φ_iY(x_i) + Σ ψ_abY(x_a, x_b))         (the outcome, and any yes/no variable)

`scale` is x itself, log x or the log-odds of x, whichever makes the equation a sum.

How it works, part by part, with a worked example: `docs/causal_kan.md`.

## Files, in reading order

`src/causal_model/` has four folders and two files. Each folder's `__init__.py` lists
what it offers, so code imports from the folder (`from src.causal_model.nodes import Node`).

| File | Class | What it does |
|---|---|---|
| **`terms/`** | | the pieces an equation is made of |
| `terms/spline.py` | `Spline` | one curve: cubic B-spline bump weights @ heights, with a band; beyond the data it continues as a straight line |
| `terms/levels.py` | `Levels` | the edge from a categorical parent: one height per level, pulled towards its neighbours (ordered) or the average (not ordered) |
| `terms/surface.py` | `Surface` | one interaction between two parents (continuous or categorical): a grid of heights holding only what depends on both |
| **`nodes/`** | | one variable's equation |
| `nodes/fitting.py` | `BayesianFitting` | how heights are chosen: a smoothness belief per curve and per surface, whose strength λ is set by the evidence |
| `nodes/scale.py` | `Scale`, `LogScale`, `LogOddsScale` | the scale an equation is a sum on: x, log x or the log-odds of x |
| `nodes/node.py` | `Node` | one continuous variable: scale(x) = constant + curves + surfaces + noise, fitted as least squares |
| `nodes/noise_shape.py` | `NoiseShape` | 15 fixed noise shapes: peaked, bell or flat; leaning right, left or neither |
| `nodes/noise_model.py` | `NoiseModel` | reads the noise's size (may grow with the prediction) and shape off the leftovers |
| `nodes/laplace.py` | `LaplaceNode` | a node with a non-bell likelihood: damped Newton and the Laplace evidence |
| `nodes/binary.py` | `BinaryNode` | a yes/no variable, the outcome Y included: the logistic model |
| `nodes/ordinal.py` | `OrdinalNode` | an ordered variable (a credit band): the ordered logit |
| `nodes/category.py` | `CategoryNode` | a variable with unordered levels (an event type): the softmax |
| `nodes/screening.py` | `PairScreening` | which pairs of parents get a surface (at most 2), decided from the data |
| `kan.py` | `CausalKAN` | the graph: nodes in causal order |
| **`discovery/`** | | finds the graph once from the training data |
| `discovery/background.py` | `Background` | what is known: the outcome, the roots and each variable's kind, and which arrows they allow |
| `discovery/node_maker.py` | `NodeMaker` | the right node for each kind of variable |
| `discovery/evidence_score.py` | `EvidenceScore` | the evidence of a variable given a parent set, on its best scale, fitted once and remembered, on the rows where no intervention set it |
| `discovery/graph.py` | `Graph` | the arrows found so far, with the loop check and the causal order |
| `discovery/graph_search.py` | `GraphSearch` | the search: the best single change (add, remove, reverse) until none helps, with a price per arrow, restarted from shaken graphs |
| `discovery/discovered_graph.py` | `DiscoveredGraph` | the result, which builds the `CausalKAN` and writes a summary |
| **`drift/`** | | did an equation change? |
| `drift/change.py` | `ChangeFit` | one fit on people from before and after, with a change part per term that is kept only if the data needs it |
| `plot.py` | `KANPlot` | one panel per curve, one heat map per surface |

Simulations, with worlds whose true equations are known:

| File | What it does |
|---|---|
| `src/simulation/world.py` | `World` and its mechanisms (`Uniform`, `Binary`, `Equation`, `Outcome`) |
| `src/simulation/scenarios.py` | six worlds, each with one drift |
| `src/simulation/drift.py` | runs each world once and saves figures and a summary in `figures/worlds/<world>/` |
| `src/simulation/repeat.py` | runs worlds over many seeds and counts how often each term is flagged |
| `src/simulator.py` | the first small world |

Benchmark for causal discovery, in `src/benchmarks/causal_arena/`: CausalArena's
public semantic SCMs (Li, Liu, Wang and Ye, 2026; huggingface.co/datasets/LAMDA-Tabular/CausalArena,
CC-BY-4.0). 50 scenarios of operational processes in 10 domains, five of them about
decisions on people (education, finance and credit, government services, healthcare,
housing), each with 17-25 named variables of mixed type and a known true graph, sampled
5 times in two regimes (1,000 observational rows; 800 observational and 200
interventional). Files are downloaded the first time a run needs them, into
`data/causal_arena/` (about 50 MB in all, ignored by git). Results go to
`figures/benchmarks/causal_arena/summary.txt` and `results.csv`.

| File | Class | What it does |
|---|---|---|
| `files.py` | `ArenaFiles` | downloads a repository file the first time it is needed, then reads it from disk |
| `scenarios.py` | `Scenario`, `ScenarioIndex` | the 50 public scenarios and their files, from the release manifest; the people domains |
| `data.py` | `ArenaData`, `ScenarioData` | one sampled file: standardised and raw values, each variable's measurement scale (from `graph.json`), the true graph, and C (which intervention produced each row) |
| `methods/base.py` | `DiscoveryMethod` | what a baseline must have: `name`, `uses_background`, `needs_change`, `find(data, roots)` |
| `methods/kairos_inputs.py` | `KairosInputs` | each variable's kind from its measurement scale, raw values for the continuous ones and codes for the others, and the intervention flags |
| `methods/kairos.py` | `KairosSearch` | kairos' `GraphSearch` with the kinds and the intervention flags |
| `methods/causal_learn.py` | `PCSearch`, `CDNODSearch` | PC; CD-NOD with C, its skeleton search also given the background knowledge; Fisher z or RCIT tests |
| `methods/cdnod_authors.py` | `AuthorsCDNODSearch` | the port of the authors' CD-NOD (below); hours per file, so not in the default run |
| `found.py` | `FoundGraph` | what a method returned; undirected pairs pointed at random (CausalArena's protocol), or the true or wrong way (best, worst) |
| `scoring.py` | `GraphScore` | directed-edge precision, recall, F1 and structural Hamming distance |
| `benchmark.py` | `ArenaBenchmark`, `Result` | one file: every method × background ("no background", the protocol; "roots given", for diagnosis) |
| `parallel.py` | `ParallelRun` | several files at once, each in its own process; keeps finished files, so a stopped run goes on |
| `report.py` | `Report` | means over files, over the people domains and by domain; failures; one CSV row per run |
| `run.py` | | the list of baselines and the command line |

CD-NOD ported from the authors' MATLAB code (Huang et al., JMLR 2020;
github.com/Biwei-Huang/Causal-Discovery-from-Nonstationary-Heterogeneous-Data), in
`src/baselines/cdnod/`: phases 1-3 of `nonsta_cd_new.m` (skeleton, invariance, and
the Independent Change Principle that causal-learn lacks), with the KCI tests, the
Gaussian-process width fitting and `minimize.m` they call. Each file names the MATLAB
file it ports and keeps its copyright notice; `meek.py` is GPL v3 like `meeks.m`,
and the CD-NOD files themselves carry no licence, so the port is for research use.
Linear algebra goes through numpy: the installed scipy's `cholesky(lower=True)` leaves
junk above the diagonal and crashes on 2,000 × 2,000 matrices.

| File | Class | MATLAB file |
|---|---|---|
| `numerics.py` | `Numerics` | `dist2`, `kernel`, `covSEard`, `eigdec`, `pdinv` |
| `null_distribution.py` | `ChiSquareMixture` | the χ² mixture at the end of the KCI tests |
| `random_features.py` | `RandomFeatures` | `transformFeatures.m` |
| `minimize.py` | `ConjugateGradients` | `minimize.m` |
| `gaussian_process.py` | `GaussianProcessWidths` | `gpr_multi.m` with `covSEard` + `covNoise` |
| `kci.py` | `KernelCITest` | `indtest_new_t.m`: `UInd_KCItest.m`, `CInd_test_new_withGP_t.m` |
| `kci_rff.py` | `RandomFeatureCITest` | `indtest_new_t_RFF.m` and the two `_RFF` tests |
| `skeleton.py` | `SkeletonSearch` | phase 1 of `nonsta_cd_new.m` |
| `meek.py` | `MeekRules` | `meeks.m` |
| `orientation.py` | `InvarianceOrientation` | phase 2 |
| `change_score.py` | `ChangeDependence` | `infer_nonsta_dir.m` |
| `independent_change.py` | `IndependentChangeOrientation` | phase 3 |
| `settings.py` | `CDNODSettings` | `pars`, and the examples' choice of test by sample size |
| `cdnod.py` | `CDNOD`, `CDNODGraphs` | `nonsta_cd_new.m` |

## Run (from `kairos/`)

    PYTHONPATH=. venv/bin/python src/simulation/drift.py
    PYTHONPATH=. venv/bin/python src/simulation/repeat.py --worlds 6 --seeds 5 --figures
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run                          # every scenario (hours; about 1 min per kairos run)
    PYTHONPATH=. venv/bin/python -m src.benchmarks.causal_arena.run --domains people --replicates 1
    bash scripts/run_causal_arena.sh                 # everything, 5 files at a time, in the background; see its header

## Tests (from `kairos/`)

    venv/bin/python -m pytest

Written as behaviour scenarios (Given / When / Then) in `tests/features/*.feature`,
with the steps in `tests/test_*.py` (pytest-bdd). They check the model against
independent references: the textbook B-spline formula, the direct evidence formula,
importance sampling for the evidence of yes/no, ordered and category nodes, a fine grid
for λ, every graph over three variables for "the score points to the right graph", and
world 1's true equations for band coverage and predictions past the data. A scenario tagged `@known_problem` describes behaviour
the model does not have yet: it is expected to fail, and pytest reports when it starts
passing. None is tagged at the moment.

## Left out on purpose

- Batch only: no running statistics, forgetting or change windows over time yet.
- Knots are evenly spaced over the data's range. Beyond it a curve continues as a
  straight line with its slope at the edge, and its band widens; `Spline.outside(x)`
  tells when a value is past the data.
- In the simulations, yes/no variables are still fitted as curves between 0 and 1 (only
  the two ends mean anything); discovery gives them, and ordered and category variables,
  their own likelihoods when `Background.kinds` says what they are.
- A node gets at most two surfaces, each over two parents. What depends on three parents
  at once (a gate on a product, say) is only partly represented.
- Ordered and category variables get no surfaces.
- No fairness policy: the causal model only describes the world as it is. A
  protected attribute can be a variable, and the curves on its edges show where
  a gap enters (with bands), but which routes are unfair is decided outside the
  model, in the recourse search and the monitor, which are not built yet.
- No counterfactuals for now: the code that answered "what if this person's income
  changed?" was removed and will return with the recourse search.
