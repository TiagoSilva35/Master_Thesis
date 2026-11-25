"""Prototype: a location-scale score. Mean curves as now (bell fit), then a log-variance
curve of the same parents, log v = W·θ, fitted by Newton on the Gaussian likelihood of the
leftovers with a bending penalty (λ by Laplace evidence). Score:
  'ls bell'   Σ log N(r_i; 0, v_i)                      + mean Occam − mean prior cost + scale Occam
  'ls learnt' Σ [log f(z_i) − ½ log v_i], z = r/√v      + the same + f's Occam"""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting, Node, NoiseDensity
from src.causal_model.nodes.fitting import GRID

def scale_fit(design, blocks, size, leftovers, ridge=1e-3):
    best = None
    r2 = leftovers ** 2
    start = np.zeros(size); start[0] = np.log(r2.mean())
    for lam in GRID:
        P = ridge * np.eye(size)
        for b in blocks:
            P[b.columns, b.columns] += lam * b.bending
        th = start.copy()
        def height(t):
            lv = design @ t
            return float(np.sum(-0.5 * lv - r2 / (2 * np.exp(lv))) - 0.5 * t @ P @ t)
        h = height(th)
        for _ in range(50):
            v = np.exp(design @ th)
            g = design.T @ (-0.5 + r2 / (2 * v)) - P @ th
            H = design.T @ (design * (r2 / (2 * v))[:, None]) + P
            step = np.linalg.solve(H, g); length = 1.0
            while True:
                trial = th + length * step; th_h = height(trial)
                if th_h >= h - 1e-9 or length < 1e-6: break
                length /= 2
            th, h = trial, th_h
            if np.max(np.abs(length * step)) < 1e-8: break
        start = th
        v = np.exp(design @ th)
        H = design.T @ (design * (r2 / (2 * v))[:, None]) + P
        occam = -0.5 * th @ P @ th + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * np.linalg.slogdet(H)[1]
        ev = float(np.sum(-0.5 * np.log(2 * np.pi * v) - r2 / (2 * v))) + occam
        if best is None or ev > best[0]:
            best = (ev, v, occam)
    return best

def score(data, child, parents, kinds, mode):
    node = Node(child, parents, pairs=[], fitting=BayesianFitting(), noise_floor=0.01, kinds=kinds)
    node.fit(data)
    r = data[child] - node.predict(data)
    if mode == "bell":
        return node.evidence - len(r) / 2 * (np.log(2 * np.pi) + 1)
    if not parents:   # a root: constant spread
        v = np.full(len(r), r.var() + 0.01 * data[child].var()); scale_occam = 0.0
    else:
        node.place_knots(data); design = node.design(data)
        _, v, scale_occam = scale_fit(design, node.smoothness_blocks(), design.shape[1], r)
        v = v + 0.01 * data[child].var()   # the floor
    base = -node.prior_cost + node.occam + scale_occam
    if mode == "ls bell":
        return float(np.sum(-0.5 * np.log(2 * np.pi * v) - r ** 2 / (2 * v))) + base
    z = r / np.sqrt(v)
    f = NoiseDensity(); f.fit(z)
    return float(np.sum(f.log_density(z) - 0.5 * np.log(v))) + base + f.occam

def margin(data, a, b, kinds, mode):
    return score(data, a, [], kinds, mode) + score(data, b, [a], kinds, mode) - score(data, b, [], kinds, mode) - score(data, a, [b], kinds, mode)

MODES = ("bell", "ls bell", "ls learnt")
if sys.argv[1] == "synthetic":
    r = np.random.default_rng(2)
    funcs = {"linear": lambda x: x, "tanh": lambda x: np.tanh(2 * x), "exp": lambda x: np.exp(x / 2)}
    spreads = {"constant": lambda x: np.ones_like(x), "grows": lambda x: 0.2 + np.abs(x), "sigmoid": lambda x: 0.2 + 1 / (1 + np.exp(-3 * x))}
    out = {m: [] for m in MODES}
    for fname, g in funcs.items():
        for sname, sp in spreads.items():
            for noise in ("normal", "uniform"):
                for seed in range(3):
                    x = r.uniform(-2, 2, 1000)
                    n = r.normal(0, 1, 1000) if noise == "normal" else r.uniform(-1.7, 1.7, 1000)
                    y = g(x) / g(x).std() + 0.5 * sp(x) * n
                    data = {"x": (x - x.mean()) / x.std(), "y": (y - y.mean()) / y.std()}
                    for m in MODES:
                        out[m].append((sname, margin(data, "x", "y", {}, m)))
    for m, rows in out.items():
        for sname in spreads:
            v = np.array([val for k, val in rows if k == sname])
            print(f"{m:10s} spread {sname:9s}: right {np.mean(v > 0):.2f}, wrong by > 3: {np.mean(v < -3):.2f}")
else:
    from pathlib import Path
    from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
    from src.benchmarks.causal_arena.methods import KairosInputs
    files = ArenaFiles(Path(sys.argv[1]))
    s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
    d = ArenaData(files).load(s, "obs_only", 0)
    inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
    edges = [(a, b) for a, b in sorted(d.truth) if kinds[a] == "continuous" and kinds[b] == "continuous"]
    res = {m: np.array([margin(table, a, b, kinds, m) for a, b in edges]) for m in MODES}
    for m in MODES:
        v = res[m]
        print(f"{sys.argv[2]:10s} {m:10s}: {len(v)} continuous edges, right {np.mean(v > 0):.2f} ({np.sum(v > 3)}✓ {np.sum(v < -3)}✗)")
