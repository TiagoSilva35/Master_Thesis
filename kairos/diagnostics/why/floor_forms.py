"""Synthetic pairs and triples: own+add (today), carried+add, carried+max (mean / row), own+max."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import Node, FlexibleNode, NoiseFloor, NoiseSpread, NoiseDensity, BayesianFitting, Fit
from src.causal_model.discovery import EvidenceScore, NodeMaker
N, SHARE = 1000, float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
MODE = {}
def solve(self, design, target, penalty):
    precision = design.T @ design + penalty
    coef = np.linalg.solve(precision, design.T @ target)
    left = np.sum((target - design @ coef) ** 2) + coef @ penalty @ coef
    floor = self.floor.rows(coef, target)
    s2 = left / len(target)
    nv = s2 + np.mean(floor) if MODE["add"] else max(s2, np.mean(floor))
    occam = 0.5 * np.linalg.slogdet(penalty)[1] - 0.5 * np.linalg.slogdet(precision)[1]
    ev = -0.5 * len(target) * np.log(nv) + occam
    if not MODE["row"]: floor = np.full(len(target), np.mean(floor))
    return Fit(coef, nv * np.linalg.inv(precision), float(ev), noise_variance=float(nv), occam=float(occam),
               prior_cost=float(coef @ penalty @ coef / (2 * nv)), floor=floor)
carried = Node.carried_floor
def floor_for(self, data): return carried(self, data) if MODE["carried"] else NoiseFloor(self.noise_floor)
def refit(self, data):
    Node.refit(self, data)
    values = data[self.name]; leftovers = values - self.predict(data)
    self.spread = NoiseSpread(self.fitting)
    self.spread.fit(self.design(data), self.spread_blocks(), leftovers)
    variance = self.spread.variance + self.row_floor if MODE["add"] else np.maximum(self.spread.variance, self.row_floor)
    z = leftovers / np.sqrt(variance)
    share = np.mean(self.row_floor) / np.mean(variance)
    width = max(share * np.sqrt(2 * np.pi * np.e), self.rounding_step(values) / np.sqrt(np.mean(variance)))
    self.density = NoiseDensity(smallest_width=width); self.density.fit(z)
    self.evidence = float(np.sum(self.density.log_density(z) - 0.5 * np.log(variance)) - self.prior_cost + self.occam + self.spread.occam + self.density.occam)
Node._solve, Node.carried_floor, FlexibleNode.refit = solve, floor_for, refit
modes = {"own+add": dict(carried=False, add=True, row=False), "own max": dict(carried=False, add=False, row=False),
         "car+add r": dict(carried=True, add=True, row=True), "car max m": dict(carried=True, add=False, row=False), "car max r": dict(carried=True, add=False, row=True)}
def score(d, learn): return EvidenceScore(d, NodeMaker({}, None, BayesianFitting(), intervals=8, noise_floor=SHARE, learn_noise=learn))
forms = {"2x": lambda x: 2*x, "exp(x)": np.exp, "x+x^3": lambda x: x + x**3, "tanh(2x)": lambda x: np.tanh(2*x), "sqrt(x+2)": lambda x: np.sqrt(np.maximum(x + 2, 0.02))}
roots = {"bell": lambda r: np.clip(r.normal(size=N), -3, 3), "flat": lambda r: r.uniform(-1.7, 1.7, N)}
print("share", SHARE, "| pairs: share of the 10 (root, formula) x 3 seeds with the true direction ahead by > 3 / behind by > 3, and mean gap")
for noise in (0.0, 0.1, 0.2, 0.4):
    for learn in (False, True):
        out = []
        for m, mode in modes.items():
            MODE.clear(); MODE.update(mode); gaps = []
            for rn, root in roots.items():
                for f in forms.values():
                    for seed in range(3):
                        r = np.random.default_rng(seed); x = root(r); y = f(x)
                        y = y + noise * np.std(y) * (r.normal(size=N) if rn == "bell" else r.uniform(-1.7, 1.7, N))
                        s = score(dict(x=x, y=y), learn)
                        gaps.append(s("x", []) + s("y", ["x"]) - s("y", []) - s("x", ["y"]))
            g = np.array(gaps); out.append(f"{m}: {np.mean(g > 3):.2f}/{np.mean(g < -3):.2f} ({g.mean():6.0f})")
        print(f"noise {noise}: {'learnt' if learn else 'bell  '} " + " | ".join(out), flush=True)
print("triples z = a x - b y: wrong minus true (worst of the two rewritings), 3 seeds, mean")
for a, b in ((1, 1), (100, 30)):
    for rn, root in roots.items():
        for learn in (False, True):
            out = []
            for m, mode in modes.items():
                MODE.clear(); MODE.update(mode); w = []
                for seed in range(3):
                    r = np.random.default_rng(seed); x, y = root(r), root(r); s = score(dict(x=x, y=y, z=a * x - b * y), learn)
                    t = s("x", []) + s("y", []) + s("z", ["x", "y"])
                    w.append(max(s("z", []) + s("y", ["z"]) + s("x", ["y", "z"]), s("z", []) + s("x", ["z"]) + s("y", ["x", "z"])) - t)
                out.append(f"{m}: {np.mean(w):6.0f}")
            print(f"{a}x-{b}y {rn} {'learnt' if learn else 'bell  '} " + " | ".join(out), flush=True)
