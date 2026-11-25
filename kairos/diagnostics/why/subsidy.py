import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.nodes import BayesianFitting, FlexibleNode
files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if "mortgage" in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
t = KairosInputs(d).table()
x = t["closing_cost_subsidy"]
vals, counts = np.unique(x, return_counts=True)
print("distinct", len(vals), "top shares", np.round(np.sort(counts)[::-1][:5] / len(x), 3), "step", FlexibleNode.rounding_step(x))
print("gaps (first 10):", np.round(np.diff(vals)[:10], 4))
node = FlexibleNode("closing_cost_subsidy", [], pairs=[], fitting=BayesianFitting(), noise_floor=0.01)
node.fit(t)
f = node.density
r = x - node.predict(t); v = node.spread.variance + 0.01 * x.var(); z = r / np.sqrt(v)
edges = f._edges(z)
print("bins", len(edges) - 1, "width", edges[1] - edges[0], "z range", z.min(), z.max(), "λ", f.smoothness)
hist = np.histogram(z, edges)[0]
print("counts (first 40):", hist[:40].tolist())
ld = f.log_density(z)
bell = -0.5 * np.log(2 * np.pi) - z ** 2 / 2
print("Σ log f(z)", ld.sum(), "Σ log bell(z)", bell.sum(), "occam f", f.occam, "evidence", node.evidence)
order = np.argsort(ld)[:5]
print("worst rows: z", np.round(z[order], 3), "log f", np.round(ld[order], 2))
