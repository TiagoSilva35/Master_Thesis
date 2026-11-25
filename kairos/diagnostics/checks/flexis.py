import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting
from src.causal_model.nodes.flexible import FlexibleNode
from tests.helpers import ImportanceSampling
for seed, noise in ((0, "laplace"), (1, "uniform"), (2, "normal")):
    r = np.random.default_rng(seed)
    n = 300
    x = r.uniform(0, 4, n)
    e = {"laplace": r.laplace(0, 0.3, n), "uniform": r.uniform(-0.5, 0.5, n), "normal": r.normal(0, 0.3, n)}[noise]
    y = np.sin(x) + e
    node = FlexibleNode("y", ["x"], pairs=[], intervals=4, fitting=BayesianFitting())
    data = {"x": x, "y": y}
    node.fit(data)
    # rebuild the chosen fit
    node.place_knots(data); D = node.design(data); blocks = node.smoothness_blocks()
    P = node.fitting.penalty([node.smoothness[b.label] for b in blocks], D.shape[1], blocks)
    fit = node._solve(D, y, P)
    f = node.density
    approx = node.evidence - f.occam
    def ll(samples):
        res = y[None, :] - samples @ D.T
        return f.log_density(res.ravel()).reshape(res.shape).sum(1)
    prior_precision = P / fit.noise_variance
    sampled = ImportanceSampling(40000).log_evidence(ll, prior_precision, fit.coefficients, fit.covariance)
    print(noise, "approx", round(approx, 2), "sampled", round(sampled, 2), "diff", round(approx - sampled, 2))
