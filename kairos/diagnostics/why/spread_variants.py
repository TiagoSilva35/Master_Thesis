import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
import src.causal_model.nodes.flexible as flex
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting, NoiseDensity, NoiseSpread, Node

class Variant(flex.FlexibleNode):
    width_rule = True
    use_spread = True
    def refit(self, data):
        Node.refit(self, data)
        values = data[self.name]; r = values - self.predict(data)
        floor = self.noise_floor * np.var(values)
        if self.use_spread:
            self.spread = NoiseSpread(self.fitting); self.spread.fit(self.design(data), self.spread_blocks(), r)
            v, occ = self.spread.variance + floor, self.spread.occam
        else:
            v, occ = np.full(len(r), np.mean(r ** 2) + floor), 0.0
        z = r / np.sqrt(v)
        smallest = max(np.sqrt(floor), self.rounding_step(values)) / np.sqrt(np.mean(v)) if self.width_rule else 0.0
        self.density = NoiseDensity(smallest_width=smallest); self.density.fit(z)
        self.evidence = float(np.sum(self.density.log_density(z) - 0.5 * np.log(v)) - self.prior_cost + self.occam + occ + self.density.occam)

def run(width_rule, use_spread, floor):
    Variant.width_rule, Variant.use_spread = width_rule, use_spread
    flex_orig = flex.FlexibleNode
    import src.causal_model.discovery.node_maker as nm
    nm.FlexibleNode = Variant
    r = np.random.default_rng(2)
    funcs = {"linear": lambda x: x, "tanh": lambda x: np.tanh(2 * x), "exp": lambda x: np.exp(x / 2)}
    spreads = {"constant": lambda x: np.ones_like(x), "grows": lambda x: 0.2 + np.abs(x), "sigmoid": lambda x: 0.2 + 1 / (1 + np.exp(-3 * x))}
    res, fails = {}, []
    for fname, g in funcs.items():
        for sname, sp in spreads.items():
            for noise in ("normal", "uniform"):
                for seed in range(3):
                    x = r.uniform(-2, 2, 1000)
                    e = r.normal(0, 1, 1000) if noise == "normal" else r.uniform(-1.7, 1.7, 1000)
                    y = g(x) / g(x).std() + 0.5 * sp(x) * e
                    data = {"x": (x - x.mean()) / x.std(), "y": (y - y.mean()) / y.std()}
                    s = EvidenceScore(data, NodeMaker({}, None, BayesianFitting(), 8, noise_floor=floor, learn_noise=True))
                    m = s("x", []) + s("y", ["x"]) - s("y", []) - s("x", ["y"])
                    res.setdefault(sname, []).append(m)
                    if m < -3: fails.append((fname, sname, noise, round(m, 1)))
    nm.FlexibleNode = flex_orig
    print(f"width rule {width_rule}, spread {use_spread}, floor {floor}: " + ", ".join(
        f"{k} {np.mean(np.array(m) > 0):.2f}" for k, m in res.items()), "| confidently wrong:", fails[:6])

for args in ((True, True, 0.01), (False, True, 0.01), (True, True, 0.0), (True, False, 0.01), (False, False, 0.01)):
    run(*args)
