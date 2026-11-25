"""Diagnostic only (nothing in the repo changes): a noise floor carried through the equation.
Every continuous variable is trusted only up to a precision of `share` of its own variance. The floor of an
equation is the child's own precision plus each parent's, times how steep its curve is, squared."""
import numpy as np
from src.causal_model.nodes.node import Node
from src.causal_model.terms import Spline
plain = Node.refit
def slope_sq(curve, x):
    h = (curve.knots[-1] - curve.knots[0]) * 1e-4 if hasattr(curve, "knots") else np.ptp(x) * 1e-4
    lo, hi = x.min(), x.max()
    a, b = np.clip(x - h, lo, hi), np.clip(x + h, lo, hi)
    return float(np.mean(((curve(b) - curve(a)) / (b - a)) ** 2))
def refit(self, data):
    if not hasattr(self, "_share"): self._share = self.noise_floor
    share, z = self._share, data[self.name]
    self.noise_floor = share
    plain(self, data)
    if share == 0 or np.var(z) == 0: return
    for _ in range(2):
        c = share * np.var(z)
        for p in self.parents:
            if isinstance(self.curves[p], Spline):
                c += share * np.var(data[p]) * slope_sq(self.curves[p], data[p])
        self.noise_floor = c / np.var(z)
        plain(self, data)
def on():
    from src.causal_model.nodes.laplace import LaplaceNode
    Node.refit = refit
    LaplaceNode.refit = plain
def off():
    Node.refit = plain
