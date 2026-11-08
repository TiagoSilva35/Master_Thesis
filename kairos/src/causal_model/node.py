"""Step 2: one variable."""

import numpy as np

from src.causal_model.fitting import Fit, Fitting, PlainFitting
from src.causal_model.spline import Spline


class Node:
    """One variable of the causal graph and its equation:

        x = constant + Σ curve_parent(x_parent) + noise

    One curve per parent (one curve per edge), summed at the node.
    A node without parents is just constant + noise.

    `fitting` decides how the constant and heights are chosen (see fitting.py).
    The default, PlainFitting, lets only the data decide.
    """

    def __init__(self, name: str, parents: list[str], intervals: int = 8,
                 fitting: Fitting | None = None):
        self.name = name
        self.parents = list(parents)
        self.intervals = intervals
        self.fitting = fitting if fitting is not None else PlainFitting()
        self.constant = 0.0
        self.curves: dict[str, Spline] = {}
        self.smoothness: dict[str, float] = {}  # λ per parent, filled by BayesianFitting

    def fit(self, data: dict[str, np.ndarray]) -> None:
        """Place the knots, then let the fitting choose the constant and every curve's heights."""
        self._place_knots(data)
        design = self._design(data)
        target = data[self.name]
        result = self.fitting.fit(lambda penalty: self._solve(design, target, penalty),
                                  design.shape[1], self._edge_columns())
        self._store(result, data)

    def _solve(self, design: np.ndarray, target: np.ndarray, penalty: np.ndarray) -> Fit:
        """For one penalty: the most believable coefficients, how unsure they are,
        and how probable the data was (the evidence).

        The noise is taken to be bell-shaped, so all three have a formula. The
        coefficients come from penalised least squares, and the noise variance
        (its average square) is estimated from what is left over.
        """
        precision = design.T @ design + penalty # penalty is the prior's precision, design.T @ design is the data's precision
        coefficients = np.linalg.solve(precision, design.T @ target)
        leftover = np.sum((target - design @ coefficients) ** 2) + coefficients @ penalty @ coefficients
        noise_variance = leftover / len(target)
        evidence = (-0.5 * len(target) * np.log(noise_variance)
                    + 0.5 * np.linalg.slogdet(penalty)[1]
                    - 0.5 * np.linalg.slogdet(precision)[1])
        return Fit(coefficients, noise_variance * np.linalg.inv(precision), float(evidence))

    def predict(self, values: dict):
        """What the parents explain: constant + Σ curves."""
        total = self.constant
        for parent, curve in self.curves.items():
            total = total + curve(values[parent])
        return total

    def noise(self, values: dict):
        """What the parents do not explain: observed value minus prediction."""
        return values[self.name] - self.predict(values)

    def _place_knots(self, data: dict[str, np.ndarray]) -> None:
        """Spread each curve's knots evenly over the range its parent takes in the data."""
        self.curves = {
            parent: Spline(data[parent].min(), data[parent].max(), self.intervals)
            for parent in self.parents
        }

    def _design(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """One row per sample: a 1 for the constant, then each curve's bump weights."""
        rows = len(data[self.name])
        columns = [np.ones((rows, 1))]
        columns += [self.curves[parent].weights(data[parent]) for parent in self.parents]
        return np.hstack(columns)

    def _edge_columns(self) -> list[slice]:
        """Which design columns hold each parent's heights (column 0 is the constant)."""
        columns, start = [], 1
        for parent in self.parents:
            size = len(self.curves[parent].heights)
            columns.append(slice(start, start + size))
            start += size
        return columns

    def _store(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """Hand the fitted numbers to the constant and the curves.

        Bump weights add up to 1, so raising a curve and lowering the constant
        by the same amount gives the same predictions. To make the split unique,
        each curve is shifted to average zero over the data and the constant
        takes the difference. Each curve also keeps how unsure its shifted
        heights are, which is what its band is drawn from.
        """
        self.constant = float(fit.coefficients[0])
        self.smoothness = dict(zip(self.parents, fit.smoothness))
        for parent, columns in zip(self.parents, self._edge_columns()):
            curve = self.curves[parent]
            heights = fit.coefficients[columns]
            average = curve.weights(data[parent]).mean(axis=0)  # curve's average = average @ heights
            shift = average @ heights
            curve.heights = heights - shift
            self.constant += float(shift)
            shifting = np.eye(len(heights)) - np.outer(np.ones(len(heights)), average)
            curve.covariance = shifting @ fit.covariance[columns, columns] @ shifting.T
