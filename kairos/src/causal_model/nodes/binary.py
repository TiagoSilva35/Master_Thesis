"""A yes/no variable."""

import numpy as np

from src.causal_model.nodes.laplace import LaplaceNode


class BinaryNode(LaplaceNode):
    """A yes/no variable, coded 0 and 1. Its equation gives a chance, not a value:

        P(x = 1) = σ(constant + Σ curve_parent(x_parent) + Σ surfaces),   σ(s) = 1 / (1 + e^-s)

    The curves are in log-odds: the logistic model. The outcome Y (1 is the
    favourable outcome) is a yes/no variable too; it only never has children
    (Background sees to that).
    """

    def predict(self, values: dict):
        """The chance that x = 1 (which is also x's expected value)."""
        return self.sigmoid(self.total(values))

    def _start_point(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.zeros(design.shape[1])

    def _log_likelihood(self, design: np.ndarray, target: np.ndarray, numbers: np.ndarray) -> float:
        """Written with logaddexp so that chances very close to 0 or 1 do not overflow."""
        score = design @ numbers
        return float(-np.sum(target * np.logaddexp(0.0, -score) + (1 - target) * np.logaddexp(0.0, score)))

    def _slope_and_curvature(self, design: np.ndarray, target: np.ndarray,
                             numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chance = self.sigmoid(design @ numbers)
        return design.T @ (target - chance), design.T @ (design * (chance * (1 - chance))[:, None])
