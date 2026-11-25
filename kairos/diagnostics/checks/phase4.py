import sys, time
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes.noise_density import NoiseDensity
from src.causal_model.nodes.flexible import FlexibleNode
from src.causal_model.nodes import Node, BayesianFitting

r = np.random.default_rng(0)
z = r.normal(0, 1, 1000)
d = NoiseDensity(); t = time.perf_counter(); d.fit(z); print("fit time", time.perf_counter() - t)
bell = np.sum(-0.5*np.log(2*np.pi*z.var()) - (z - z.mean())**2/(2*z.var()))
print("bell numbers: λ", d.smoothness, "loglik f", d.log_density(z).sum(), "bell", bell, "occam", d.occam)
u = r.uniform(-1, 1, 1000)
d.fit(u); bellu = np.sum(-0.5*np.log(2*np.pi*u.var()) - (u - u.mean())**2/(2*u.var()))
print("uniform: λ", d.smoothness, "loglik f", d.log_density(u).sum(), "bell", bellu, "flat f at ±0.5:", np.exp(d.log_density(np.array([-0.5, 0, 0.5]))))
# flexible with bell-ish data equals bell evidence minus constant?
x = r.uniform(0, 4, 500); y = np.sin(x) + r.normal(0, 0.3, 500)
a = Node("y", ["x"], pairs=[], fitting=BayesianFitting()); a.fit({"x": x, "y": y})
b = FlexibleNode("y", ["x"], pairs=[], fitting=BayesianFitting()); b.fit({"x": x, "y": y})
print("bell node", a.evidence - 250*(np.log(2*np.pi)+1), "flexible", b.evidence, "density occam", b.density.occam, "λ", b.density.smoothness)
# direction: straight line, uniform cause and noise
def total(data, forward):
    names = ("x", "y") if forward else ("y", "x")
    out = {}
    for kind in (Node, FlexibleNode):
        root = kind(names[0], [], pairs=[], fitting=BayesianFitting()); root.fit(data)
        child = kind(names[1], [names[0]], pairs=[], fitting=BayesianFitting()); child.fit(data)
        out[kind.__name__] = root.evidence + child.evidence
    return out
x = r.uniform(-1, 1, 1000); y = 2*x + r.uniform(-1, 1, 1000)
f, b2 = total({"x": x, "y": y}, True), total({"x": x, "y": y}, False)
print("line forward", f, "reverse", b2)
x = r.uniform(0, 1, 1000); y = x**2 + r.normal(0, 0.05, 1000)
f, b2 = total({"x": x, "y": y}, True), total({"x": x, "y": y}, False)
print("x^2 forward", f, "reverse", b2)
