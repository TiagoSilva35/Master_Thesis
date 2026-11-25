import sys, time
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.discovery import Background, Graph, GraphSearch, OrderSearch

def total(score, graph):
    return sum(score(v, list(graph.parents[v])) for v in graph.variables)

# 1. collider traps with bell scoring
for seed in range(8):
    r = np.random.default_rng(seed)
    x, y = r.normal(0, 1, 500), r.normal(0, 1, 500)
    data = {"x": x, "y": y, "z": x + y + r.normal(0, 0.5, 500)}
    search = GraphSearch(Background())
    score = search.score(data)
    g = Graph(sorted(data)); search.add_arrows(g, score)
    better = OrderSearch().improve(g, score, Background())
    print(f"collider seed {seed}: greedy {sorted(g.arrows())} {total(score, g):.1f} | order {sorted(better.arrows())} {total(score, better):.1f}")

# 2. random 8-variable nonlinear graphs, learnt noise
def random_scm(seed, n=800, p=8):
    r = np.random.default_rng(seed)
    order = list(range(p)); truth = set(); data = {}
    for j in order:
        parents = [i for i in range(j) if r.uniform() < 0.35]
        noise = r.uniform(-1, 1, n) if r.uniform() < 0.5 else r.laplace(0, 0.5, n)
        value = sum(np.tanh(r.uniform(0.5, 1.5) * data[f"v{i}"]) * r.choice([-1, 1]) for i in parents) + noise
        data[f"v{j}"] = (value - value.mean()) / value.std()
        truth |= {(f"v{i}", f"v{j}") for i in parents}
    return data, truth
for seed in range(6):
    data, truth = random_scm(seed)
    t = time.perf_counter()
    search = GraphSearch(Background(), noise_floor=0.01, learn_noise=True)
    score = search.score(data)
    g = Graph(sorted(data)); search.add_arrows(g, score); greedy_total = total(score, g); t1 = time.perf_counter() - t
    better = OrderSearch().improve(g, score, Background()); t2 = time.perf_counter() - t - t1
    def f1(graph):
        found = set(graph.arrows()); right = len(found & truth)
        return 2 * right / (len(found) + len(truth)) if found or truth else 1.0
    print(f"random seed {seed}: greedy total {greedy_total:9.1f} F1 {f1(g):.2f} ({t1:.1f}s) | order total {total(score, better):9.1f} F1 {f1(better):.2f} ({t2:.1f}s)  truth total {sum(score(v, sorted(a for a, b in truth if b == v)) for v in data):9.1f}")
