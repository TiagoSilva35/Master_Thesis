"""Greedy's graph vs the true graph vs the best pruned true graph (each variable keeps its best subset of true parents)."""
import sys, json, time
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from itertools import combinations
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex, GraphScore
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import Background, Graph, GraphSearch
files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
learn = sys.argv[3] == "learnt"
d = ArenaData(files).load(s, "obs_only", 0)
nodes = {n["id"]: n for n in json.loads(files.path(s.graph_file).read_text())["nodes"]}
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
search = GraphSearch(Background(kinds=kinds), noise_floor=0.01, learn_noise=learn)
score = search.score(table)
t = time.perf_counter()
g = Graph(sorted(table)); search.add_arrows(g, score)
truth = {v: sorted(a for a, b in d.truth if b == v) for v in d.variables}
def best_subset(v):
    ps = truth[v]
    if len(ps) <= 5:
        return max((list(c) for k in range(len(ps) + 1) for c in combinations(ps, k)), key=lambda c: score(v, c))
    cur = list(ps)
    while cur:
        c = max(([p for p in cur if p != q] for q in cur), key=lambda c: score(v, c))
        if score(v, c) <= score(v, cur): break
        cur = c
    return cur
pruned = {v: best_subset(v) for v in d.variables}
greedy = {v: list(g.parents[v]) for v in d.variables}
tot = lambda parents: sum(score(v, parents[v]) for v in d.variables)
arrows = lambda parents: {(a, b) for b, ps in parents.items() for a in ps}
print(f"=== {sys.argv[2]} {sys.argv[3]}  N={len(next(iter(table.values())))}  variables={len(d.variables)}  ({time.perf_counter()-t:.0f}s)")
for name, p in (("greedy", greedy), ("true, all arrows", truth), ("true, pruned", pruned)):
    print(f"  {name:18s} total {tot(p):10.1f}  arrows {len(arrows(p)):3d}  F1 {GraphScore.compare(arrows(p), d.truth).f1:.2f}")
def unexplained(v, ps):
    if kinds[v] != "continuous" or v in score.constant: return float("nan")
    node = score.maker.make(v, [p for p in sorted(ps) if p not in score.constant]); node.fit(table)
    return float(np.var(node.noise(table)) / np.var(table[v]))
rows = sorted(((score(v, greedy[v]) - score(v, pruned[v]), v) for v in d.variables), reverse=True)
for gap, v in rows:
    print(f"  {gap:8.1f} {v[:34]:34s} {kinds[v][:5]} unexpl greedy {unexplained(v, greedy[v]):6.3f} true {unexplained(v, truth[v]):6.3f} | greedy {[p[:20] for p in greedy[v]]} | true {[p[:20] for p in truth[v]]} kept {[p[:20] for p in pruned[v]]}")
