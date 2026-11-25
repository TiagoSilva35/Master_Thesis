import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos"); sys.path.insert(0, sys.argv[1])
import numpy as np, floor_patch
from src.causal_model.discovery import Background, GraphSearch
rng = np.random.default_rng(0)
N = 1000
def totals(data, learn):
    sc = GraphSearch(Background(kinds={k: "continuous" for k in data}), noise_floor=0.01, learn_noise=learn).score(data)
    true = sc("x", []) + sc("y", []) + sc("z", ["x", "y"])
    wx = sc("z", []) + sc("y", ["z"]) + sc("x", ["y", "z"])
    wy = sc("z", []) + sc("x", ["z"]) + sc("y", ["x", "z"])
    return true, wx, wy
cases = {}
for noise in ("bell", "flat"):
    draw = (lambda: rng.normal(size=N)) if noise == "bell" else (lambda: rng.uniform(-1.7, 1.7, N))
    x, y = draw(), draw(); cases[f"{noise} z=x-y"] = dict(x=x, y=y, z=x - y)
    x, y = draw(), draw(); cases[f"{noise} z=100x-30y"] = dict(x=x, y=y, z=100 * x - 30 * y)
    x, y = draw(), draw(); cases[f"{noise} z=x-y+noise(10%var)"] = dict(x=x, y=y, z=x - y + rng.normal(size=N) * np.sqrt(2 * .1))
for name, data in cases.items():
    rho2 = np.corrcoef(data["y"], data["z"])[0, 1] ** 2; rx = np.corrcoef(data["x"], data["z"])[0, 1] ** 2
    pred = (-N / 2 * np.log(1 - rho2), -N / 2 * np.log(1 - rx))
    for learn in (False, True):
        for label, switch in (("now", floor_patch.off), ("carried", floor_patch.on)):
            switch(); t, wx, wy = totals(data, learn)
            print(f"{name:26s} {'learnt' if learn else 'bell  '} floor {label:8s}: wrong − true  x←y,z {wx - t:8.1f}   y←x,z {wy - t:8.1f}   (predicted for 'now', exact bell: {pred[0]:.0f}, {pred[1]:.0f})")
