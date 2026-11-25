"""Curved formulas, exact and noisy: true minus reverse under three floors for the learnt score, and for the bell."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import Node, FlexibleNode, NoiseFloor
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting
N = 1000
row_version = FlexibleNode.refit; carried = Node.carried_floor; keep = Node.store
def own_only(self, data): return NoiseFloor(self.noise_floor)
def store_mean(self, fit, data):
    keep(self, fit, data); self.row_floor = np.full(1, np.mean(fit.floor)) if len(fit.floor) > 1 else fit.floor
modes = {"own": (own_only, keep), "mean": (carried, store_mean), "row": (carried, keep)}
forms = {"2x": lambda x: 2*x, "exp(x)": np.exp, "x+x^3": lambda x: x + x**3, "tanh(2x)": lambda x: np.tanh(2*x), "sqrt(x+2)": lambda x: np.sqrt(np.maximum(x + 2, 0.02))}
roots = {"bell": lambda r: np.clip(r.normal(size=N), -3, 3), "flat": lambda r: r.uniform(-1.7, 1.7, N)}
for noise in (0.0, 0.1, 0.4):
    print(f"--- noise spread = {noise} x spread of f(x); numbers are true minus reverse, 3 seeds averaged")
    for rn, root in roots.items():
        for fn, f in forms.items():
            row = {}
            for m, (cf, st) in modes.items():
                Node.carried_floor, Node.store = cf, st
                for learn in (False, True):
                    v = []
                    for seed in range(3):
                        r = np.random.default_rng(seed); x = root(r); y = f(x)
                        y = y + noise * np.std(y) * (r.normal(size=N) if rn == "bell" else r.uniform(-1.7, 1.7, N))
                        d = dict(x=x, y=y)
                        s = EvidenceScore(d, NodeMaker({}, None, BayesianFitting(), intervals=8, noise_floor=0.01, learn_noise=learn))
                        v.append(s("x", []) + s("y", ["x"]) - s("y", []) - s("x", ["y"]))
                    row[m, learn] = np.mean(v)
            print(f"{rn} root {fn:10s} bell: own {row['own',False]:8.1f} carried {row['row',False]:8.1f} | learnt: own {row['own',True]:8.1f} mean {row['mean',True]:8.1f} row {row['row',True]:8.1f}")
