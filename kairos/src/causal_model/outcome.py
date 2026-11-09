"""Step 5: the outcome Y."""

import numpy as np

from src.causal_model.fitting import Fit, Fitting
from src.causal_model.node import Node


class OutcomeNode(Node):
    """The outcome Y: 1 is the favourable outcome, 0 is not.

    Y is a node like any other, but its equation gives a chance, not a value:

        P(Y = 1) = σ(constant + Σ curve_parent(x_parent) + Σ surfaces),   σ(s) = 1 / (1 + e^-s)

    The curves are in log-odds. Least squares cannot fit this, so `_solve` takes
    Newton steps on the log-likelihood instead.
    """

    def __init__(self, name: str, parents: list[str], pairs: list[tuple[str, str]], intervals: int = 8,
                 steps: int = 25, fitting: Fitting | None = None,
                 pair_intervals: int = 6,
                 screening=None):
        super().__init__(name, parents, pairs, intervals, pair_intervals, fitting, screening)
        self.steps = steps
        self._start = None  # where Newton starts: the previous solution while λ is being searched

    def solver(self, design: np.ndarray, target: np.ndarray):
        self._start = None  # a new design: Newton starts from zero again
        return super().solver(design, target)

    def _solve(self, design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> Fit:
        """For one penalty: Newton's method for the most believable coefficients,
        then the Laplace approximation for how unsure they are and the evidence.

        The chance is not bell-shaped, so there is no formula. Newton climbs to
        the top of the hill (at most `steps` steps); near its top the hill is
        treated as a bell whose width comes from the curvature there. A step is
        kept only if it climbs; otherwise it is halved until it does (without
        this, nearly duplicate columns can make Newton overshoot and run away).
        While the fitting tries one λ after another the hill barely moves, so
        Newton starts from the previous top instead of from zero.
        """
        coefficients = np.zeros(design.shape[1]) if self._start is None else self._start.copy()
        height = self._height(design, target, penalty, coefficients)
        for _ in range(self.steps):
            chance = self._sigmoid(design @ coefficients)
            slope = design.T @ (target - chance) - penalty @ coefficients
            step = np.linalg.solve(self._curvature(design, chance, penalty), slope)
            length = 1.0
            while True:
                trial = coefficients + length * step
                trial_height = self._height(design, target, penalty, trial)
                if trial_height >= height - 1e-9 or length < 1e-6:
                    break
                length /= 2
            coefficients, height = trial, trial_height
            if np.max(np.abs(length * step)) < 1e-8:
                break
        self._start = coefficients
        chance = self._sigmoid(design @ coefficients)
        curvature = self._curvature(design, chance, penalty)
        evidence = (height + 0.5 * np.linalg.slogdet(penalty)[1] - 0.5 * np.linalg.slogdet(curvature)[1])
        return Fit(coefficients, np.linalg.inv(curvature), float(evidence))

    @staticmethod
    def _height(design, target, penalty, coefficients) -> float:
        """The hill Newton climbs: the log-likelihood minus the penalty. Written
        with logaddexp so that chances very close to 0 or 1 do not overflow."""
        score = design @ coefficients
        fit_to_data = -np.sum(target * np.logaddexp(0.0, -score) + (1 - target) * np.logaddexp(0.0, score))
        return float(fit_to_data - 0.5 * coefficients @ penalty @ coefficients)

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
