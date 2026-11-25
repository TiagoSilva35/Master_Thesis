import sys, json
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
files = ArenaFiles(Path(sys.argv[1]))
for word in ("mortgage", "medication"):
    s = next(x for x in ScenarioIndex.load(files) if word in x.slug)
    d = ArenaData(files).load(s, "obs_int", 0)
    graph = json.loads(files.path(s.graph_file).read_text())
    nodes = {n["id"]: n for n in graph["nodes"]}
    print(word, "| intervention_targets:", json.dumps(graph.get("intervention_targets"))[:300])
    for j, v in enumerate(d.variables):
        rows = d.intervened[:, j]
        if not rows.any(): continue
        parents = sorted(a for a, b in d.truth if b == v)
        children = sorted(b for a, b in d.truth if a == v)
        obs, setv = d.raw[~d.intervened.any(axis=1), j], d.raw[rows, j]
        print(f"  {v}: {rows.sum()} rows | true parents {parents} | children {len(children)}")
        print(f"     observational mean {obs.mean():.3g} sd {obs.std():.3g} | set rows mean {setv.mean():.3g} sd {setv.std():.3g} | distinct set values {np.unique(np.round(setv, 6))[:6]}")
        print(f"     semantics: {str(nodes[v].get('intervention_semantics'))[:250]}")
