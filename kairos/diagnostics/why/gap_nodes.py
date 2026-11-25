import sys, json
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import Background, Graph, GraphSearch
files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if "mortgage" in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
nodes = {n["id"]: n for n in json.loads(files.path(s.graph_file).read_text())["nodes"]}
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
search = GraphSearch(Background(kinds=kinds), noise_floor=0.01, learn_noise=True)
score = search.score(table)
g = Graph(sorted(table)); search.add_arrows(g, score); search.prune(g, score)
truth = {v: sorted(a for a, b in d.truth if b == v) for v in d.variables}
rows = sorted(((score(v, list(g.parents[v])) - score(v, truth[v]), v) for v in d.variables), reverse=True)
for gap, v in rows[:7]:
    print(f"{v} ({kinds[v]}): found parents beat true parents by {gap:.0f}")
    print(f"    true parents:  {truth[v]}")
    print(f"    found parents: {sorted(g.parents[v])}")
    print(f"    true equation: {str(nodes[v]['structural_mechanism'].get('equation_plan', ''))[:220]}")
