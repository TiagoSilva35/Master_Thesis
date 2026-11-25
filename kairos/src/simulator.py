"""Simulate a small world, fit a causal KAN to it, and plot its curves.

Run from kairos/:  PYTHONPATH=. venv/bin/python src/simulator.py
"""

from pathlib import Path

import numpy as np

from src.causal_model.kan import CausalKAN
from src.causal_model.nodes import BinaryNode, Node
from src.causal_model.plot import KANPlot

FIGURES = Path(__file__).resolve().parents[1] / "figures"


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


def build_model() -> CausalKAN:
    """The simulator's graph."""
    return CausalKAN([
        Node("income", [], intervals=16),
        Node("savings", ["income"], intervals=16),
        BinaryNode("Y", ["income", "savings"], intervals=16),
    ])


def main() -> None:
    model = build_model()
    model.fit(Simulator(seed=2).sample(10000))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "kan_curves.png"
    KANPlot(model).figure().savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
