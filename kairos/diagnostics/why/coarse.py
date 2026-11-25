import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes import BayesianFitting, FlexibleNode, Node

r = np.random.default_rng(0)
n = 1000
x = r.normal(0, 1, n)
print(f"{'R²':>5} {'floor':>6} {'Σlog f − Σlog bell':>20} {'bins':>5} {'width/σr':>9}")
for r2 in (0.3, 0.75, 0.9, 0.97):
    noise = np.sqrt((1 - r2) / r2)
    y = x + r.normal(0, noise, n)
    for floor in (0.0, 0.01):
        node = FlexibleNode("y", ["x"], pairs=[], fitting=BayesianFitting(), noise_floor=floor)
        node.fit({"x": x, "y": y})
        res = y - node.predict({"x": x, "y": y})
        bell = np.sum(-0.5 * np.log(2 * np.pi * res.var()) - res ** 2 / (2 * res.var()))
        f = node.density
        knots = f.spline.knots
        width = (knots[-1] - knots[0]) / (len(knots) - 1) / 5  # bins: 100 over the grid, 17 intervals
        print(f"{r2:5.2f} {floor:6.2f} {f.log_density(res).sum() - bell:20.1f} "
              f"{'':>5} {(knots[-1]-knots[0])/res.std():9.1f} (grid span in leftover spreads)")
