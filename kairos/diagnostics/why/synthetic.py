"""Direction accuracy on synthetic additive-noise pairs: x → y = g(x) + noise, both standardised."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting

r = np.random.default_rng(1)
funcs = {"linear": lambda x: x, "square": lambda x: x ** 2, "tanh": lambda x: np.tanh(2 * x), "exp": lambda x: np.exp(x)}
causes = {"uniform": lambda n: r.uniform(-1.7, 1.7, n), "normal": lambda n: r.normal(0, 1, n), "skewed": lambda n: r.exponential(1, n)}
noises = {"uniform": lambda n: r.uniform(-1, 1, n), "normal": lambda n: r.normal(0, 1, n), "laplace": lambda n: r.laplace(0, 1, n)}
configs = {"bell": dict(noise_floor=0.01), "learnt+floor": dict(noise_floor=0.01, learn_noise=True),
           "learnt no floor": dict(noise_floor=0.0, learn_noise=True)}
results = {name: [] for name in configs}
for fname, g in funcs.items():
    for cname, cause in causes.items():
        for nname, noise in noises.items():
            for strength in (0.3, 1.0):   # noise share: high R² vs low R²
                n = 1000
                x = cause(n); gx = g(x); gx = (gx - gx.mean()) / gx.std()
                y = gx + strength * noise(n) / noise(100000).std()
                data = {"x": (x - x.mean()) / x.std(), "y": (y - y.mean()) / y.std()}
                line = f"{fname:7s} {cname:8s} {nname:8s} s={strength}:"
                for name, kw in configs.items():
                    s = EvidenceScore(data, NodeMaker({}, None, BayesianFitting(), 8, **kw))
                    m = s("x", []) + s("y", ["x"]) - s("y", []) - s("x", ["y"])
                    results[name].append((m, fname, cname, nname, strength))
                    line += f"  {name} {m:8.1f}"
                print(line)
print()
for name, rows in results.items():
    m = np.array([row[0] for row in rows])
    gaussian = np.array([row[1] == "linear" and row[2] == "normal" and row[3] == "normal" for row in rows])
    print(f"{name:16s} right {np.mean(m[~gaussian] > 0):.2f}, right by > 3: {np.mean(m[~gaussian] > 3):.2f}, "
          f"wrong by > 3: {np.mean(m[~gaussian] < -3):.2f}  (of {np.sum(~gaussian)} identifiable pairs)")
