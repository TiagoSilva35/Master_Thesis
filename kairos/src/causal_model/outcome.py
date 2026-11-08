"""Step 5: the outcome Y."""

import numpy as np

from src.causal_model.fitting import Fit, Fitting
from src.causal_model.node import Node


class OutcomeNode(Node):
    """The outcome Y: 1 is the favourable outcome, 0 is not.

    Y is a node like any other, but its equation gives a chance, not a value:

        P(Y = 1) = σ(constant + Σ curve_parent(x_parent)),   σ(s) = 1 / (1 + e^-s)

    The curves are in log-odds. Least squares cannot fit this, so `_solve` takes
    Newton steps on the log-likelihood instead.
    """

    def __init__(self, name: str, parents: list[str], intervals: int = 8,
                 steps: int = 25, fitting: Fitting | None = None):
        super().__init__(name, parents, intervals, fitting)
        self.steps = steps

    def _solve(self, design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> Fit:
        """For one penalty: Newton's method for the most believable coefficients,
        then the Laplace approximation for how unsure they are and the evidence.

        The chance is not bell-shaped, so there is no formula. Newton climbs to
        the top of the hill (at most `steps` steps); near its top the hill is
        treated as a bell whose width comes from the curvature there.
        """
        coefficients = np.zeros(design.shape[1])
        for _ in range(self.steps):
            chance = self._sigmoid(design @ coefficients)
            slope = design.T @ (target - chance) - penalty @ coefficients
            step = np.linalg.solve(self._curvature(design, chance, penalty), slope)
            coefficients = coefficients + step
            if np.max(np.abs(step)) < 1e-8:
                break
        chance = np.clip(self._sigmoid(design @ coefficients), 1e-12, 1 - 1e-12)
        curvature = self._curvature(design, chance, penalty)
        fit_to_data = np.sum(target * np.log(chance) + (1 - target) * np.log(1 - chance))
        evidence = (fit_to_data - 0.5 * coefficients @ penalty @ coefficients
                    + 0.5 * np.linalg.slogdet(penalty)[1]
                    - 0.5 * np.linalg.slogdet(curvature)[1])
        return Fit(coefficients, np.linalg.inv(curvature), float(evidence))

    def predict(self, values: dict):
        """The chance that Y = 1 (which is also Y's expected value)."""
        return self._sigmoid(super().predict(values))

    def noise(self, values: dict):
        """Y is not known for the person we ask about, so there is no noise to
        recover. Zero makes `predict + noise` give the chance itself."""
        return 0.0

    @staticmethod
    def _curvature(design: np.ndarray, chance: np.ndarray, penalty: np.ndarray) -> np.ndarray:
        """How sharply the log-likelihood (plus penalty) bends at these coefficients."""
        return design.T @ (design * (chance * (1 - chance))[:, None]) + penalty

    @staticmethod
    def _sigmoid(score):
        return 1.0 / (1.0 + np.exp(-score))
