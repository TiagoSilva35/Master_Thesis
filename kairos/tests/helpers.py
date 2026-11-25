"""Independent references the tests compare the model against."""

import numpy as np

from src.benchmarks.causal_arena.found import FoundGraph
from src.causal_model.kan import CausalKAN
from src.causal_model.nodes import BayesianFitting, BinaryNode, Node
from src.simulation.scenarios import SCENARIOS


class EmptyMethod:
    """A benchmark method that finds no arrows. Here, so that another process can import it."""

    name, uses_background, needs_change = "empty", True, False

    def find(self, data, roots):
        return FoundGraph(set(), set(), len(data.values), 0.0)


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
                       BinaryNode("Y", ["income", "savings"], pairs=[], intervals=16, fitting=fitting)])
    model.fit(people)
    return model


class ImportanceSampling:
    """log ∫ likelihood(numbers) × prior(numbers) d numbers, estimated by drawing
    numbers from a bell twice as wide as Laplace's around its top and weighting
    each draw. Independent of the Laplace code: it needs only the log-likelihood
    of each draw. The prior is the bell with precision `penalty` around 0."""

    def __init__(self, draws: int, seed: int = 2):
        self.draws = draws
        self.seed = seed

    def log_evidence(self, log_likelihood, penalty: np.ndarray, top: np.ndarray, covariance: np.ndarray) -> float:
        random = np.random.default_rng(self.seed)
        spread = 2.0 * covariance
        samples = random.multivariate_normal(top, spread, self.draws)
        size = len(top)
        log_prior = (0.5 * np.linalg.slogdet(penalty)[1] - size / 2 * np.log(2 * np.pi)
                     - 0.5 * np.einsum("ij,jk,ik->i", samples, penalty, samples))
        offset = samples - top
        log_proposal = (-0.5 * np.einsum("ij,jk,ik->i", offset, np.linalg.inv(spread), offset)
                        - 0.5 * np.linalg.slogdet(spread)[1] - size / 2 * np.log(2 * np.pi))
        log_weights = log_likelihood(samples) + log_prior - log_proposal
        return float(np.log(np.mean(np.exp(log_weights - log_weights.max()))) + log_weights.max())
