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
inp = KairosInputs(d); t, kinds = inp.table(), inp.kinds
child = sys.argv[2] if len(sys.argv) > 2 else "final_closing_costs"
parent = sys.argv[3] if len(sys.argv) > 3 else "credit_score_band"
x = t[child]
print(child, "distinct", len(np.unique(x)), "step", FlexibleNode.rounding_step(x), "| parent", parent, kinds[parent], np.unique(t[parent], return_counts=True))
for parents in ([], [parent]):
    n = FlexibleNode(child, parents, pairs=[], fitting=BayesianFitting(), noise_floor=0.01, kinds=kinds)
    n.fit(t)
    r = x - n.predict(t); v = n.spread.variance + 0.01 * x.var(); z = r / np.sqrt(v)
    f = n.density
    print(f"parents {parents}: Σlog f {f.log_density(z).sum():8.1f}  −½Σlog v {-0.5*np.log(v).sum():8.1f}  prior {-n.prior_cost:7.1f}  occam {n.occam:7.1f}  "
          f"spread occam {n.spread.occam:7.1f}  f occam {f.occam:7.1f}  total {n.evidence:8.1f} | bins {len(f.edges)-1} width {f.edges[1]-f.edges[0]:.4f} λf {f.smoothness} "
          f"| r var {r.var():.4f} v range {v.min():.4f}-{v.max():.4f} | bell Σlog N(z) {np.sum(-0.5*np.log(2*np.pi) - z**2/2):8.1f}")
