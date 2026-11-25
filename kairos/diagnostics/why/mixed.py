"""Mixed true edges: the four pieces of the direction margin under three scorings."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
import src.causal_model.discovery.node_maker as nm
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting, FlexibleNode, Node, NoiseDensity

class NoSpread(FlexibleNode):
    def refit(self, data):
        Node.refit(self, data)
        values = data[self.name]; r = values - self.predict(data)
        floor = self.noise_floor * np.var(values)
        v = np.full(len(r), np.mean(r ** 2) + floor)
        z = r / np.sqrt(v)
        self.density = NoiseDensity(smallest_width=self._smallest_width(values, floor, v)); self.density.fit(z)
        self.evidence = float(np.sum(self.density.log_density(z) - 0.5 * np.log(v)) - self.prior_cost + self.occam + self.density.occam)

files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
fit = BayesianFitting()
class NoSpreadMaker(NodeMaker):
    def make(self, name, parents, screening=None, for_scoring=True):
        node = super().make(name, parents, screening, for_scoring)
        if isinstance(node, FlexibleNode):
            node.__class__ = NoSpread
        return node

scores = {"bell": EvidenceScore(table, NodeMaker(kinds, None, fit, 8, noise_floor=0.01))}
scores["shape"] = EvidenceScore(table, NoSpreadMaker(kinds, None, fit, 8, noise_floor=0.01, learn_noise=True))
scores["spread+shape"] = EvidenceScore(table, NodeMaker(kinds, None, fit, 8, noise_floor=0.01, learn_noise=True))
cont = lambda v: kinds[v] == "continuous"
print(f"{'edge (true direction)':62s} {'scoring':13s} {'s(a)':>9s} {'s(b|a)':>9s} {'s(b)':>9s} {'s(a|b)':>9s} {'margin':>8s}")
for a, b in sorted(d.truth):
    if cont(a) == cont(b):
        continue
    for name, sc in scores.items():
        sa, sba, sb, sab = sc(a, []), sc(b, [a]), sc(b, []), sc(a, [b])
        label = f"{a[:26]}({kinds[a][:3]}) → {b[:26]}({kinds[b][:3]})" if name == "bell" else ""
        print(f"{label:62s} {name:13s} {sa:9.1f} {sba:9.1f} {sb:9.1f} {sab:9.1f} {sa + sba - sb - sab:8.1f}")
