"""Did the order search change anything on the smoke files? Totals and arrows before pruning."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import Background, Graph, GraphSearch, OrderSearch, ParentSets
files = ArenaFiles(Path(sys.argv[1]))
s = next(x for x in ScenarioIndex.load(files) if sys.argv[2] in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inputs = KairosInputs(d); table, kinds = inputs.table(), inputs.kinds
background = Background(kinds=kinds)
search = GraphSearch(background, noise_floor=0.01, learn_noise=True)
score = search.score(table)
greedy = Graph(sorted(table)); search.add_arrows(greedy, score)
total = lambda g: sum(score(v, list(g.parents[v])) for v in g.variables)
order_search = OrderSearch()
sets = ParentSets(score, background, order_search.candidates, order_search.most)
sets.prepare(greedy.variables, {v: list(greedy.parents[v]) for v in greedy.variables})
start = greedy.order()
first = order_search.graph(start, sets)
better = order_search.improve(greedy, score, background)
print(f"{sys.argv[2]}: greedy total {total(greedy):.1f} | best sets in greedy's order {total(first):.1f} | after order search {total(better):.1f}")
print("  arrows changed by the order search:", sorted(set(better.arrows()) ^ set(greedy.arrows()))[:12])
big = [(v, len(greedy.parents[v])) for v in greedy.variables if len(greedy.parents[v]) > order_search.most]
print("  variables with more than 4 parents (old cap) in greedy's graph:", big)
