"""Independent references the tests compare the model against."""

import numpy as np

from src.causal_model.fitting import BayesianFitting
from src.causal_model.kan import CausalKAN
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.simulation.scenarios import SCENARIOS


def sigmoid(score):
    return 1.0 / (1.0 + np.exp(-score))


class CoxDeBoor:
    """The textbook B-spline basis (the Cox-de Boor recursion), written without
    using the Spline class."""

    def __init__(self, knots: np.ndarray, degree: int = 3):
        self.knots = knots
        self.degree = degree

    def weights(self, x: np.ndarray) -> np.ndarray:
        knots, degree = self.knots, self.degree
        basis = np.array([(knots[i] <= x) & (x < knots[i + 1]) for i in range(len(knots) - 1)], float).T
        basis[x == knots[-degree - 1], len(knots) - degree - 2] = 1.0  # the right end of the range
        for d in range(1, degree + 1):
            new = np.zeros((len(x), len(knots) - d - 1))
            for i in range(len(knots) - d - 1):
                left = (x - knots[i]) / (knots[i + d] - knots[i]) * basis[:, i]
                right = (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1]) * basis[:, i + 1]
                new[:, i] = left + right
            basis = new
        return basis


def world(index: int):
    """A drift scenario from scenarios.py, by position (0 is world 1)."""
    return SCENARIOS[index]()


def world_one_model(people: dict, ridge: float = 1e-3) -> CausalKAN:
    fitting = BayesianFitting(ridge=ridge)
    model = CausalKAN([Node("income", [], pairs=[], intervals=16, fitting=fitting),
                       Node("savings", ["income"], pairs=[], intervals=16, fitting=fitting),
                       OutcomeNode("Y", ["income", "savings"], pairs=[], intervals=16, fitting=fitting)])
    model.fit(people)
    return model
