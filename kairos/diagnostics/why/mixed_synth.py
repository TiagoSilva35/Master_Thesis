"""Synthetic mixed pairs: margins for the true direction under bell and learnt scoring."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting

def margin(data, a, b, kinds, learn):
    s = EvidenceScore(data, NodeMaker(kinds, None, BayesianFitting(), 8, noise_floor=0.01, learn_noise=learn))
    return s(a, []) + s(b, [a]) - s(b, []) - s(a, [b])

def sig(t): return 1 / (1 + np.exp(-t))
cases = {}
for seed in range(3):
    r = np.random.default_rng(seed); n = 1000
    a = r.uniform(-2, 2, n); cases.setdefault("uniform a → yes/no b (logistic)", []).append(({"a": a, "b": (r.uniform(size=n) < sig(2 * a)).astype(float)}, "a", "b", {"b": "binary"}))
    a = r.exponential(1, n); cases.setdefault("skewed a → yes/no b (threshold + 10% flips)", []).append(({"a": a, "b": np.where(r.uniform(size=n) < 0.1, a <= 1, a > 1).astype(float)}, "a", "b", {"b": "binary"}))
    a = r.normal(0, 1, n); cases.setdefault("bell a → 3 ordered levels (ordered logit)", []).append(({"a": a, "b": np.digitize(1.5 * a + r.logistic(size=n), [-1, 1]).astype(float)}, "a", "b", {"b": "ordinal"}))
    b = r.choice(3, n, p=[0.3, 0.5, 0.2]).astype(float); cases.setdefault("3 levels b → a = means + bell noise", []).append(({"b": b, "a": np.array([0, 1, 2.5])[b.astype(int)] + r.normal(0, 0.7, n)}, "b", "a", {"b": "nominal"}))
    b = r.choice(3, n, p=[0.3, 0.5, 0.2]).astype(float); cases.setdefault("3 levels b → a = means + uniform noise, spread per level", []).append(({"b": b, "a": np.array([0, 1, 2.5])[b.astype(int)] + np.array([0.3, 0.8, 1.5])[b.astype(int)] * r.uniform(-1, 1, n)}, "b", "a", {"b": "nominal"}))
    b = (r.uniform(size=n) < 0.3).astype(float); c = r.choice([0, 1, 2], n).astype(float)
    cases.setdefault("yes/no b → a, with a hidden 3-valued second cause", []).append(({"b": b, "a": 1.0 * b + 2.0 * c + r.normal(0, 0.3, n)}, "b", "a", {"b": "binary"}))
for name, items in cases.items():
    bell = [margin(d, x, y, k, False) for d, x, y, k in items]
    learnt = [margin(d, x, y, k, True) for d, x, y, k in items]
    print(f"{name:58s} bell {np.round(bell, 1)}   learnt {np.round(learnt, 1)}")
