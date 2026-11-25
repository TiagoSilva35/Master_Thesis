"""How far a node's values stray from its prediction, as curves of its parents."""

import numpy as np

from src.causal_model.nodes.fitting import BayesianFitting, Fit, Fitting, SmoothnessBlock


class NoiseSpread:
    """The noise's spread as curves of the parents (the location-scale noise model,
    Immer et al. 2023):

        log v = c + Σ g_parent(x_parent),   v = spread²

    with one curve per parent, built from the same bumps as the node's mean curves.
    It is fitted to the leftovers r (value − prediction): at each parent value, r²
    averages about v. In practice Newton climbs

        Σ [−½ log v_i − r_i² / (2 v_i)] − ½ numbers · penalty · numbers

    (the bell-shaped likelihood of the leftovers, concave in the numbers, so there
    is a single top), and the Laplace approximation gives the evidence.

    The penalty is on first differences (flattening), so a strong pull makes every
    curve flat: one spread for everyone, the plain node's model. λ per curve is
    chosen by the evidence through the fitting, so a curve only bends where many
    people show a consistent change in spread. Nothing is estimated from one row.

    After `fit`: `variance` is each row's v, and `occam` is what the curves' freedom
    costs, −½ numbers·P·numbers + ½ log|P| − ½ log|H|.
    """

    def __init__(self, fitting: Fitting | None = None, steps: int = 50):
        self.fitting = fitting if fitting is not None else BayesianFitting()
        self.steps = steps
        self.variance = np.zeros(0)
        self.occam = 0.0
        self.smoothness: list[float] = []

    def fit(self, design: np.ndarray, blocks: list[SmoothnessBlock], leftovers: np.ndarray) -> None:
        """`design`: the node's design (a 1 for the constant, then each parent's bump
        weights); `blocks`: one flattening block per parent's columns."""
        squares = leftovers ** 2
        start = np.zeros(design.shape[1])
        start[0] = np.log(max(squares.mean(), 1e-300))
        self._start = start
        fit = self.fitting.fit(lambda penalty: self._solve(design, squares, penalty), design.shape[1], blocks)
        self.variance = np.exp(design @ fit.coefficients)
        self.occam, self.smoothness = fit.occam, fit.smoothness

    def _solve(self, design: np.ndarray, squares: np.ndarray, penalty: np.ndarray) -> Fit:
        numbers = self._start.copy()
        height = self._height(design, squares, penalty, numbers)
        for _ in range(self.steps):
            variance = np.exp(design @ numbers)
            slope = design.T @ (-0.5 + squares / (2 * variance)) - penalty @ numbers
            curvature = design.T @ (design * (squares / (2 * variance))[:, None]) + penalty
            step = np.linalg.solve(curvature, slope)
            length = 1.0
            while True:
                trial = numbers + length * step
                trial_height = self._height(design, squares, penalty, trial)
                if trial_height >= height - 1e-9 or length < 1e-6:
                    break
                length /= 2
            numbers, height = trial, trial_height
            if np.max(np.abs(length * step)) < 1e-8:
                break
        self._start = numbers  # while λ is searched the top barely moves
        variance = np.exp(design @ numbers)
        curvature = design.T @ (design * (squares / (2 * variance))[:, None]) + penalty
        occam = (-0.5 * numbers @ penalty @ numbers + 0.5 * np.linalg.slogdet(penalty)[1]
                 - 0.5 * np.linalg.slogdet(curvature)[1])
        evidence = float(np.sum(-0.5 * np.log(2 * np.pi * variance) - squares / (2 * variance))) + occam
        return Fit(numbers, np.linalg.inv(curvature), evidence, occam=float(occam))

    @staticmethod
    def _height(design: np.ndarray, squares: np.ndarray, penalty: np.ndarray, numbers: np.ndarray) -> float:
        log_variance = design @ numbers
        with np.errstate(over="ignore"):
            fit_to_data = np.sum(-0.5 * log_variance - squares * np.exp(-log_variance) / 2)
        return float(fit_to_data - 0.5 * numbers @ penalty @ numbers)
