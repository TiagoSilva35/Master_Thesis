import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
import numpy as np
from src.causal_model.nodes.noise_density import NoiseDensity
for seed in range(5):
    z = np.random.default_rng(seed).normal(0, 1, 1000)
    d = NoiseDensity(); d.fit(z)
    bell = np.sum(-0.5*np.log(2*np.pi*z.var()) - (z - z.mean())**2/(2*z.var()))
    # truncated bell on the bin range
    print(seed, "λ", d.smoothness, "f - bell", round(d.log_density(z).sum() - bell, 2), "occam", round(d.occam, 2))
