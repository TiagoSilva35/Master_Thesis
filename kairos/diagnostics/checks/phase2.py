import sys, time
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting, OrdinalNode, CategoryNode, BinaryNode

def importance(node, design, target, penalty, fit, loglik, draws=60000, widen=2.0, seed=2):
    r = np.random.default_rng(seed)
    spread = widen * fit.covariance
    samples = r.multivariate_normal(fit.coefficients, spread, draws)
    ll = loglik(samples)
    size = len(fit.coefficients)
    lp = 0.5*np.linalg.slogdet(penalty)[1] - size/2*np.log(2*np.pi) - 0.5*np.einsum("ij,jk,ik->i", samples, penalty, samples)
    off = samples - fit.coefficients
    lq = -0.5*np.einsum("ij,jk,ik->i", off, np.linalg.inv(spread), off) - 0.5*np.linalg.slogdet(spread)[1] - size/2*np.log(2*np.pi)
    w = ll + lp - lq
    return np.log(np.mean(np.exp(w - w.max()))) + w.max()

def sig(s): return 1/(1+np.exp(-s))

r = np.random.default_rng(3)
n = 300
x = r.uniform(-2, 2, n)
# ordinal: 3 levels
s = 1.5*x
u = r.logistic(size=n)
y = np.digitize(s + u, [-0.5, 1.0]).astype(float)
node = OrdinalNode("y", ["x"], pairs=[], intervals=4, fitting=BayesianFitting(), log_odds_spread=5.0)
data = {"x": x, "y": y}
node.place_knots(data); design = node.design(data); solve = node.solver(design, y)
blocks = node.smoothness_blocks()
for lam in (1.0, 100.0):
    pen = node.fitting.penalty([lam]*len(blocks), node.size(design), blocks)
    node._start = None
    fit = solve(pen)
    codes = np.searchsorted(node.levels, y).astype(int)
    p = design.shape[1]
    def ll(samples):
        sc = samples[:, :p] @ design.T
        edges = np.column_stack([np.full(len(samples), -np.inf), np.zeros(len(samples)), samples[:, p:], np.full(len(samples), np.inf)])
        up = np.take_along_axis(edges, np.broadcast_to(codes+1, sc.shape), 1) - sc
        lo = np.take_along_axis(edges, np.broadcast_to(codes, sc.shape), 1) - sc
        prob = sig(up) - sig(lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.all(prob > 0, axis=1), np.sum(np.log(np.maximum(prob, 1e-300)), axis=1), -np.inf)
    full_pen = node._with_belief(pen)
    # slope check
    slope, curv = node._slope_and_curvature(design, codes.astype(float), fit.coefficients)
    print("ordinal λ", lam, "evidence", fit.evidence, "IS", importance(node, design, y, full_pen, fit, ll), "slope", np.abs(slope - full_pen @ fit.coefficients).max())
    # numeric gradient check
    eps = 1e-6; th = fit.coefficients + 0.1
    g, H = node._slope_and_curvature(design, codes.astype(float), th)
    ng = np.array([(node._log_likelihood(design, codes.astype(float), th + eps*e) - node._log_likelihood(design, codes.astype(float), th - eps*e))/(2*eps) for e in np.eye(len(th))])
    nH = np.array([(node._slope_and_curvature(design, codes.astype(float), th + eps*e)[0] - node._slope_and_curvature(design, codes.astype(float), th - eps*e)[0])/(2*eps) for e in np.eye(len(th))])
    print("  grad err", np.abs(g - ng).max(), "hess err", np.abs(-H - nH).max())

# category: 3 categories
scores = np.column_stack([np.zeros(n), 1.0*x, -1.0*x + 0.5])
pr = np.exp(scores); pr /= pr.sum(1, keepdims=True)
c = np.array([r.choice([0, 1, 2], p=row) for row in pr]).astype(float)
c = np.where(c == 0, 5.0, c)  # codes 1, 2, 5
node = CategoryNode("c", ["x"], pairs=[], intervals=4, fitting=BayesianFitting(), log_odds_spread=5.0)
data = {"x": x, "c": c}
node.place_knots(data); design = node.design(data); solve = node.solver(design, c)
blocks = node.smoothness_blocks()
print("levels", node.levels, "reference", node.reference, "others", node.others, "size", node.size(design))
for lam in (1.0, 100.0):
    pen = node.fitting.penalty([lam]*len(blocks), node.size(design), blocks)
    node._start = None
    fit = solve(pen)
    which = np.searchsorted(node.others, c); which[c == node.reference] = -1
    def ll(samples):
        out = np.zeros(len(samples))
        B = samples[:, node._layout]  # draws × (k-1) × p
        S = np.einsum("dkp,np->dnk", B, design)
        full = np.concatenate([np.zeros(S.shape[:2] + (1,)), S], axis=2)
        chosen = np.take_along_axis(full, np.broadcast_to((which + 1)[None, :, None], (len(samples), n, 1)), 2)[..., 0]
        return np.sum(chosen - np.logaddexp.reduce(full, axis=2), axis=1)
    full_pen = node._with_belief(pen)
    slope, curv = node._slope_and_curvature(design, which.astype(float), fit.coefficients)
    print("category λ", lam, "evidence", fit.evidence, "IS", importance(node, design, c, full_pen, fit, ll, draws=20000), "slope", np.abs(slope - full_pen @ fit.coefficients).max())
    th = fit.coefficients + 0.1
    g, H = node._slope_and_curvature(design, which.astype(float), th)
    ng = np.array([(node._log_likelihood(design, which.astype(float), th + eps*e) - node._log_likelihood(design, which.astype(float), th - eps*e))/(2*eps) for e in np.eye(len(th))])
    nH = np.array([(node._slope_and_curvature(design, which.astype(float), th + eps*e)[0] - node._slope_and_curvature(design, which.astype(float), th - eps*e)[0])/(2*eps) for e in np.eye(len(th))])
    print("  grad err", np.abs(g - ng).max(), "hess err", np.abs(-H - nH).max())

# full fits + renumbering
node1 = CategoryNode("c", ["x"], pairs=[], fitting=BayesianFitting(), log_odds_spread=5.0); node1.fit({"x": x, "c": c})
c2 = np.select([c == 1, c == 2, c == 5], [7.0, 0.0, 3.0])
node2 = CategoryNode("c", ["x"], pairs=[], fitting=BayesianFitting(), log_odds_spread=5.0); node2.fit({"x": x, "c": c2})
print("renumber", node1.evidence, node2.evidence)
t = time.perf_counter(); o = OrdinalNode("y", ["x"], pairs=[], fitting=BayesianFitting(), log_odds_spread=5.0); o.fit({"x": x, "y": y}); print("ordinal fit", o.evidence, o.cut_points, time.perf_counter()-t)
# separation
b = (x > 0).astype(float)
bn = BinaryNode("b", ["x"], pairs=[], fitting=BayesianFitting(), log_odds_spread=5.0); bn.fit({"x": x, "b": b}); print("separation", bn.evidence, np.abs(bn.covariance).max(), bn.constant, max(np.abs(cv.heights).max() for cv in bn.curves.values()))
