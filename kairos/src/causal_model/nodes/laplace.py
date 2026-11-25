"""A node whose likelihood has no formula for the evidence."""

import numpy as np

from src.causal_model.nodes.fitting import Fit, Solve, log_det
from src.causal_model.nodes.node import Node


class LaplaceNode(Node):
    """A node whose values are not bell-shaped around its prediction: a yes/no
    variable, an ordered level, a category. Its numbers cannot come from least
    squares, so `_solve` climbs the hill

        height(numbers) = log-likelihood(numbers) − ½ numbers · penalty · numbers

    with Newton's method, and the Laplace approximation turns the top of the
    hill into the evidence: near its top the hill is treated as a bell whose
    width comes from the curvature there.

    The hill has a single top. The log-likelihoods of the logistic, ordered
    logit and softmax models are concave in the scores, the scores are weighted
    sums of the numbers, and the penalty is an upside-down bowl; a sum of
    concave pieces is concave. So Newton reaches the top from any start.

    Subclasses give only the likelihood: `_start_point`, `_log_likelihood` and
    `_slope_and_curvature` (its first derivative, and minus its second).

    The fitting's ridge is a belief about log-odds here. NodeMaker uses a ridge of
    1/5², a weak belief that log-odds beyond ±10 (probabilities beyond 0.99995) are
    rare. It keeps a category that is (almost) always drawn from pushing its
    log-odds towards infinity.
    """

    steps = 25                         # Newton steps at most
    _start: np.ndarray | None = None   # where Newton starts: the previous top while λ is being searched

    def solver(self, design: np.ndarray, target: np.ndarray) -> Solve:
        self._start = None  # a new design: Newton starts from the subclass's start point again
        return lambda penalty: self._solve(design, target, penalty)

    def noise(self, values: dict):
        """A category's noise is not defined, so there is none to recover. Zero
        makes `predict + noise` give the prediction itself."""
        return 0.0

    def read_noise(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """No noise, so nothing to read."""

    def leftovers(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Observed value minus expected value."""
        return (data[self.name] - self.predict(data))[:, None]

    def _solve(self, design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> Fit:
        """For one penalty: Newton's method for the most believable numbers,
        then the Laplace approximation for how unsure they are and the evidence.

        A step is kept only if it climbs; otherwise it is halved until it does
        (without this, nearly duplicate columns can make Newton overshoot). A
        step that leaves the allowed region (cut-points out of order) has height
        −∞ and is halved too. While the fitting tries one λ after another the
        hill barely moves, so Newton starts from the previous top.
        """
        numbers = self._start_point(design, target) if self._start is None else self._start.copy()
        height = self._height(design, target, penalty, numbers)
        for _ in range(self.steps):
            slope, curvature = self._slope_and_curvature(design, target, numbers)
            step = np.linalg.solve(curvature + penalty, slope - penalty @ numbers)
            length = 1.0
            while True:
                trial = numbers + length * step
                trial_height = self._height(design, target, penalty, trial)
                if trial_height >= height - 1e-9 or length < 1e-6:
                    break
                length /= 2
            if not np.isfinite(trial_height):  # even the shortest step left the allowed region
                break
            numbers, height = trial, trial_height
            if np.max(np.abs(length * step)) < 1e-8:
                break
        self._start = numbers
        curvature = self._slope_and_curvature(design, target, numbers)[1] + penalty
        evidence = height + 0.5 * log_det(penalty) - 0.5 * log_det(curvature)
        return Fit(numbers, np.linalg.inv(curvature), float(evidence))

    def _height(self, design: np.ndarray, target: np.ndarray, penalty: np.ndarray, numbers: np.ndarray) -> float:
        """The hill Newton climbs: the log-likelihood minus the penalty."""
        return float(self._log_likelihood(design, target, numbers) - 0.5 * numbers @ penalty @ numbers)

    def _start_point(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _log_likelihood(self, design: np.ndarray, target: np.ndarray, numbers: np.ndarray) -> float:
        raise NotImplementedError

    def _slope_and_curvature(self, design: np.ndarray, target: np.ndarray,
                             numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def sigmoid(score):
        """σ(s) = 1 / (1 + e^-s), written with tanh so that it never overflows."""
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(score, dtype=float)))
