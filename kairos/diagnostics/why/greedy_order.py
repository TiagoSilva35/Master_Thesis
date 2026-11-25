"""Where does greedy go wrong? Log each added arrow (true / reversed / extra) with its gain,
and at the end, which true arrows were blocked because they would close a loop."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import Background, Graph, GraphSearch

files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
search = GraphSearch(Background(kinds=kinds), noise_floor=0.01, learn_noise=True)
score = search.score(table)
graph = Graph(sorted(table))
gains = {}
for child in graph.variables:
    search._update_gains(gains, graph, score, child)
step = 0
log = []
while True:
    best = search._best_arrow(gains, graph)
    if best is None or gains[best] <= 0:
        break
    parent, child = best
    kind = "true" if best in d.truth else ("REVERSED" if (child, parent) in d.truth else "extra")
    log.append((step, kind, parent, child, gains[best]))
    graph.add(parent, child)
    search._update_gains(gains, graph, score, child)
    step += 1
for step, kind, parent, child, gain in log[:40]:
    print(f"{step:3d} {kind:9s} {parent[:30]:30s} → {child[:30]:30s} gain {gain:8.1f}")
found = set(graph.arrows())
missing = [a for a in d.truth if a not in found and (a[1], a[0]) not in found]
blocked = [a for a in missing if graph.creates_loop(*a)]
print(f"\n{len(log)} arrows added: {sum(k == 'true' for _, k, *_ in log)} true, {sum(k == 'REVERSED' for _, k, *_ in log)} reversed, {sum(k == 'extra' for _, k, *_ in log)} extra")
print(f"true arrows missing: {len(missing)}, of which blocked by a loop at the end: {len(blocked)}")
for a in blocked:
    print("   blocked:", a, " standalone gain:", round(score(a[1], [a[0]]) - score(a[1], []), 1))
rev = [(p, c) for _, k, p, c, _ in log if k == "REVERSED"]
print("reversed arrows, with the gain of flipping each in the final graph (positive = flip helps):")
for p, c in rev:
    if (p, c) in found:
        flip = (score(c, [x for x in graph.parents[c] if x != p]) + score(p, graph.parents[p] + [c])) - (score(c, graph.parents[c]) + score(p, graph.parents[p]))
        print(f"   {p[:30]} → {c[:30]}: {flip:8.1f}{'  (would close a loop)' if graph.reversal_creates_loop(p, c) else ''}")
