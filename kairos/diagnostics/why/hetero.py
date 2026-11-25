"""Synthetic pairs where the noise's spread grows with the cause: y = g(x) + s(x)·n."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting
r = np.random.default_rng(2)
funcs = {"linear": lambda x: x, "tanh": lambda x: np.tanh(2 * x), "exp": lambda x: np.exp(x / 2)}
spreads = {"constant": lambda x: np.ones_like(x), "grows": lambda x: 0.2 + np.abs(x), "sigmoid": lambda x: 0.2 + 1 / (1 + np.exp(-3 * x))}
out = {"bell": [], "learnt": []}
for fname, g in funcs.items():
    for sname, sp in spreads.items():
        for noise in ("normal", "uniform"):
            for seed in range(3):
                x = r.uniform(-2, 2, 1000)
                n = r.normal(0, 1, 1000) if noise == "normal" else r.uniform(-1.7, 1.7, 1000)
                y = g(x) / g(x).std() + 0.5 * sp(x) * n
                data = {"x": (x - x.mean()) / x.std(), "y": (y - y.mean()) / y.std()}
                for name, kw in (("bell", {}), ("learnt", {"learn_noise": True})):
                    s = EvidenceScore(data, NodeMaker({}, None, BayesianFitting(), 8, noise_floor=0.01, **kw))
                    out[name].append((sname, s("x", []) + s("y", ["x"]) - s("y", []) - s("x", ["y"])))
for name, rows in out.items():
    for sname in spreads:
        m = np.array([v for k, v in rows if k == sname])
        print(f"{name:7s} spread {sname:9s}: right {np.mean(m > 0):.2f}, wrong by > 3: {np.mean(m < -3):.2f}")
