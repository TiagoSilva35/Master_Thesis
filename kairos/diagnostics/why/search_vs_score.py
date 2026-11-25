"""Is it the search or the score? Total score of the found graph vs the true graph,
and the true graph after pruning (true arrows whose removal raises the score)."""
import sys, time
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex, GraphScore
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import Background, GraphSearch

files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
truth_parents = {v: sorted(a for a, b in d.truth if b == v) for v in d.variables}
for learn in (False, True):
    search = GraphSearch(Background(kinds=kinds), noise_floor=0.01, learn_noise=learn)
    score = search.score(table)
    t = time.perf_counter()
    graph = search.graph = None
    from src.causal_model.discovery import Graph
    g = Graph(sorted(table)); search.add_arrows(g, score); gains = search.prune(g, score)
    found = {v: list(g.parents[v]) for v in d.variables}
    total = lambda parents: sum(score(v, parents[v]) for v in d.variables)
    s_found, s_true = total(found), total(truth_parents)
    found_set = {(a, b) for b, ps in found.items() for a in ps}
    sc = GraphScore.compare(found_set, d.truth)
    print(f"{sys.argv[2]} {'learnt' if learn else 'bell  '}: found graph {s_found:10.1f} (F1 {sc.f1:.2f}, {len(found_set)} arrows) | true graph {s_true:10.1f} | true − found {s_true - s_found:8.1f} | {time.perf_counter() - t:.0f}s")
    # per node: where does the truth beat the found parents?
    rows = sorted(((score(v, truth_parents[v]) - score(v, found[v]), v) for v in d.variables))
    print("   nodes where the found parents score better than the true ones (top 5):",
          [(v[:24], round(x, 1)) for x, v in rows[:5]])
    print("   nodes where the true parents score better (top 5):",
          [(v[:24], round(x, 1)) for x, v in rows[::-1][:5]])
