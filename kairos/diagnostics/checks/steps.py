import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.nodes import FlexibleNode
files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if "mortgage" in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inp = KairosInputs(d); t = inp.table()
print({k: sum(1 for v in inp.kinds.values() if v == k) for k in ("continuous", "binary", "ordinal", "nominal")})
for name, values in t.items():
    if inp.kinds[name] == "continuous":
        step = FlexibleNode.rounding_step(values)
        if step: print(f"{name:40s} step {step:.5f}  floor scale {np.sqrt(0.01*values.var()):.3f}")
