import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting, Node

def fit(g, y, kind):
    n = Node("y", ["g"], pairs=[], fitting=BayesianFitting(), kinds={"g": kind}); n.fit({"g": g, "y": y}); return n

for seed in range(6):
    r = np.random.default_rng(seed)
    g = r.integers(0, 8, 200).astype(float)
    y = 0.25 * g + r.normal(0, 1, 200)
    perm = np.array([3, 6, 0, 5, 1, 7, 2, 4], float)
    a, b = fit(g, y, "ordinal").evidence, fit(perm[g.astype(int)], y, "ordinal").evidence
    print(seed, "ordered true", round(a, 2), "shuffled", round(b, 2), "diff", round(a - b, 2), "nominal", round(fit(g, y, "nominal").evidence, 2))
r = np.random.default_rng(0)
g = r.integers(0, 5, 500).astype(float); y = np.array([0, 0, 0, 1, 1.0])[g.astype(int)] + r.normal(0, 0.5, 500)
n = fit(g, y, "ordinal"); print("step heights", np.round(n.curves["g"].heights, 3), n.smoothness)
b = r.integers(0, 2, 400).astype(float); y = 0.8 * b + r.normal(0, 0.5, 400)
n = fit(b, y, "binary"); h = n.curves["g"].heights; print("binary", h, h[1]-h[0])
