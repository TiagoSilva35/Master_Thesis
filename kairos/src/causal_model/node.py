"""Step 2: one variable."""

import numpy as np

from src.causal_model.fitting import Fit, Fitting, PlainFitting, SmoothnessBlock
from src.causal_model.spline import Spline
from src.causal_model.surface import Surface


class Node:
    """One variable of the causal graph and its equation:

        x = constant + Σ curve_parent(x_parent) + Σ surface_pair(x_a, x_b) + noise

    One curve per parent (one curve per edge), summed at the node. `pairs` adds
    a surface for each listed pair of parents: the part of x that depends on
    both together (an interaction). A node without parents is just constant + noise.

    `fitting` decides how the constant, curves and surfaces are chosen (see
    fitting.py). The default, PlainFitting, lets only the data decide.
    `screening` (a PairScreening, see screening.py) chooses the pairs from the
    data instead; without it the node keeps `pairs` as given.
    """

    def __init__(self, name: str, 
                 parents: list[str], 
                 pairs: list[tuple[str, str]], 
                 intervals: int = 8,
                 pair_intervals: int = 6,
                 fitting: Fitting | None = None,
                 screening=None):
        self.name = name
        self.parents = list(parents)
        self.intervals = intervals
        self.fitting = fitting if fitting is not None else PlainFitting()
        self.pairs = [(first, second) for first, second in pairs]
        self.pair_intervals = pair_intervals
        self.screening = screening
        self._check_pairs()
        self.constant = 0.0
        self.curves: dict[str, Spline] = {}
        self.surfaces: dict[tuple[str, str], Surface] = {}
        self.smoothness: dict[str, float] = {}  # λ per smoothness block, filled by BayesianFitting
        self.covariance = None                  # how unsure all stored heights are, together
        self.rows: dict[str, slice] = {}        # which rows of `covariance` belong to each term
        self.evidence = None                    # how probable the data was under the chosen fit

    def fit(self, data: dict[str, np.ndarray]) -> None:
        """Choose the pairs (if a screening is set), then fit."""
        if self.screening is not None:
            self.pairs = self.screening.choose(self, data)
        self.refit(data)

    def refit(self, data: dict[str, np.ndarray]) -> None:
        """Fit with the current pairs: place the knots, then let the fitting choose
        the constant, curves and surfaces."""
        self.place_knots(data)
        design = self.design(data)
        result = self.fitting.fit(self.solver(design, data[self.name]), design.shape[1], self.smoothness_blocks())
        self.store(result, data)

    def predict(self, values: dict):
        """What the parents explain: constant + Σ curves + Σ surfaces."""
        total = self.constant
        for parent, curve in self.curves.items():
            total = total + curve(values[parent])
        for (first, second), surface in self.surfaces.items():
            total = total + surface(values[first], values[second])
        return total

    def noise(self, values: dict):
        """What the parents do not explain: observed value minus prediction."""
        return values[self.name] - self.predict(values)

    def terms(self) -> dict[str, Spline | Surface]:
        """Every curve and surface, by label: a curve is named after its parent,
        a surface after its pair, like "income×savings"."""
        terms: dict[str, Spline | Surface] = dict(self.curves)
        for pair, surface in self.surfaces.items():
            terms[self.surface_label(pair)] = surface
        return terms

    @staticmethod
    def surface_label(pair: tuple[str, str]) -> str:
        return f"{pair[0]}×{pair[1]}"

    # The steps of a fit. They are public because ChangeFit (change.py) runs
    # them itself, on people from two windows at once.

    def place_knots(self, data: dict[str, np.ndarray]) -> None:
        """Spread each curve's knots evenly over the range its parent takes in the
        data; each surface does the same in both directions, with fewer intervals."""
        self.curves = {
            parent: Spline(data[parent].min(), data[parent].max(), self.intervals)
            for parent in self.parents
        }
        self.surfaces = {}
        for first, second in self.pairs:
            surface = Surface(Spline(data[first].min(), data[first].max(), self.pair_intervals),
                              Spline(data[second].min(), data[second].max(), self.pair_intervals))
            surface.centre_on(data[first], data[second])
            self.surfaces[(first, second)] = surface

    def design(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """One row per sample: a 1 for the constant, then each curve's bump weights,
        then each surface's columns."""
        rows = len(data[self.name])
        columns = [np.ones((rows, 1))]
        columns += [self.curves[parent].weights(data[parent]) for parent in self.parents]
        columns += [self.surfaces[pair].design(data[pair[0]], data[pair[1]]) for pair in self.pairs]
        return np.hstack(columns)

    def columns(self) -> dict[str, slice]:
        """Which design columns hold each term's numbers (column 0 is the constant)."""
        columns, start = {}, 1
        for parent in self.parents:
            size = len(self.curves[parent].heights)
            columns[parent] = slice(start, start + size)
            start += size
        for pair in self.pairs:
            size = self.surfaces[pair].free.shape[1]
            columns[self.surface_label(pair)] = slice(start, start + size)
            start += size
        return columns

    def smoothness_blocks(self) -> list[SmoothnessBlock]:
        """One smoothness belief per curve, two per surface (one along each parent)."""
        columns = self.columns()
        blocks = [SmoothnessBlock(parent, columns[parent], self.curves[parent].bending())
                  for parent in self.parents]
        for pair in self.pairs:
            label = self.surface_label(pair)
            along_first, along_second = self.surfaces[pair].bending()
            blocks.append(SmoothnessBlock(f"{label} along {pair[0]}", columns[label], along_first))
            blocks.append(SmoothnessBlock(f"{label} along {pair[1]}", columns[label], along_second))
        return blocks

    def solver(self, design: np.ndarray, target: np.ndarray):
        """A solve function for a fitting: hand it a penalty, get back a Fit."""
        return lambda penalty: self._solve(design, target, penalty)

    def store(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """Hand the fitted numbers to the constant, the curves and the surfaces.

        Bump weights add up to 1, so raising a curve and lowering the constant
        by the same amount gives the same predictions. To make the split unique,
        each curve is shifted to average zero over the data and the constant
        takes the difference. A surface's free numbers are turned into its grid.
        """
        self.constant = float(fit.coefficients[0])
        self.evidence = fit.evidence
        self.smoothness = {block.label: strength
                           for block, strength in zip(self.smoothness_blocks(), fit.smoothness)}
        columns = self.columns()
        pieces = {}  # label -> matrix that turns the term's fitted numbers into its stored heights
        for parent in self.parents:
            curve = self.curves[parent]
            average = curve.weights(data[parent]).mean(axis=0)  # curve's average = average @ heights
            self.constant += float(average @ fit.coefficients[columns[parent]])
            pieces[parent] = np.eye(len(curve.heights)) - np.outer(np.ones(len(curve.heights)), average)
        for pair in self.pairs:
            pieces[self.surface_label(pair)] = self.surfaces[pair].free
        self._keep(pieces, columns, fit)

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

    def _keep(self, pieces: dict[str, np.ndarray], columns: dict[str, slice], fit: Fit) -> None:
        """Give each term its heights and its own spread, and keep how unsure all
        heights are together (curves and surfaces of one node move together)."""
        size = sum(matrix.shape[0] for matrix in pieces.values())
        transform = np.zeros((size, len(fit.coefficients)))
        self.rows, start = {}, 0
        for label, matrix in pieces.items():
            self.rows[label] = slice(start, start + matrix.shape[0])
            transform[self.rows[label], columns[label]] = matrix
            start += matrix.shape[0]
        heights = transform @ fit.coefficients
        self.covariance = transform @ fit.covariance @ transform.T
        for label, term in self.terms().items():
            term.heights = heights[self.rows[label]]
            term.covariance = self.covariance[self.rows[label], self.rows[label]]

    def _check_pairs(self) -> None:
        for pair in self.pairs:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(self.parents):
                raise ValueError(f"{self.name}: a pair must be two different parents, got {pair}")
