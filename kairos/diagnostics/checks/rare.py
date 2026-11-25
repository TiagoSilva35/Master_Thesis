import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting, BinaryNode
from tests.helpers import ImportanceSampling, sigmoid
r = np.random.default_rng(4); x = r.uniform(-2, 2, 400); y = (r.uniform(size=400) < sigmoid(-4.2 + 0.5*x)).astype(float)
node = BinaryNode("y", ["x"], pairs=[], intervals=4, fitting=BayesianFitting(), log_odds_spread=5.0)
d = {"x": x, "y": y}; node.place_knots(d); D = node.design(d); solve = node.solver(D, y); b = node.smoothness_blocks()
for lam in (10.0,):
    P = node.fitting.penalty([lam]*len(b), node.size(D), b); f = solve(P); Pf = node._with_belief(P)
    def ll(s):
        sc = s @ D.T; return -(y*np.logaddexp(0,-sc) + (1-y)*np.logaddexp(0, sc)).sum(1)
    print("ones", y.sum(), "laplace", f.evidence, "sampled", ImportanceSampling(60000).log_evidence(ll, Pf, f.coefficients, f.covariance))
