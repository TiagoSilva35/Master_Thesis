"""Simulate a small world, fit a causal KAN to it, and plot its curves.

Run from the repo root:  python3 -m kairos.simulator
"""

import numpy as np
import os
from src.causal_model.fitting import BayesianFitting, Fitting, PlainFitting
from src.causal_model.kan import CausalKAN
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.causal_model.plot import KANPlot


class Simulator:
    # The true world:
    #   income  = uniform between 0 and 4
    #   savings = 0.5 * income**2 + noise
    #   Y       = 1 with chance σ(-4 + income + 0.5 * savings)
    #   savings -> Y = 0.5 * savings in log-odds, so 1 unit of savings is worth 0.5 units of income.
    #   income -> Y = 1 * income in log-odds, so 1 unit of income is worth 2 units of savings.

    def __init__(self, seed: int = 0, noise: float = 0.3):
        self.random = np.random.default_rng(seed)
        self.noise = noise

    def sample(self, size: int) -> dict[str, np.ndarray]:
        income = self.random.uniform(0.0, 4.0, size)
        savings = 0.5 * income**2 + self.random.normal(0.0, self.noise, size)
        chance = 1.0 / (1.0 + np.exp(-(-4.0 + income + 0.5 * savings)))
        y = (self.random.uniform(size=size) < chance).astype(float)
        return {"income": income, "savings": savings, "Y": y}


def build_model(fitting: Fitting, intervals: int) -> CausalKAN:
    """The simulator's graph, with every node fitted the same way."""
    return CausalKAN([
        Node("income", [], intervals, fitting),
        Node("savings", ["income"], intervals, fitting),
        OutcomeNode("Y", ["income", "savings"], intervals, fitting=fitting),
    ])


def main() -> None:
    data = Simulator(seed=2).sample(10000)
    # Plain: only the data decides, so few knots keep the curves from wiggling.
    # Bayesian: the smoothness belief keeps them calm, so more knots can be used.
    models = {
        "plain": build_model(PlainFitting(), intervals=8),
        "bayesian": build_model(BayesianFitting(), intervals=16),
    }
    # create a directory to save the figures
    directory = "kairos/figures"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, model in models.items():
        model.fit(data)
        KANPlot(model).figure().savefig(f"{directory}/kan_curves_{name}.png", dpi=110)
        print(f"saved kan_curves_{name}.png")


if __name__ == "__main__":
    main()
