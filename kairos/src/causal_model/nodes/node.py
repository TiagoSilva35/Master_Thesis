"""Step 2: one variable."""

import numpy as np

from src.causal_model.nodes.fitting import BayesianFitting, Fit, SmoothnessBlock, Solve, log_det
from src.causal_model.nodes.noise_model import NoiseModel
from src.causal_model.nodes.scale import Scale
from src.causal_model.nodes.screening import PairScreening
from src.causal_model.terms import Levels, Spline, Surface


class Node:
    """One variable of the causal graph and its equation:

        scale(x) = constant + Σ curve_parent(x_parent) + Σ surface_pair(x_a, x_b) + noise

    One curve per parent (one curve per edge), summed at the node. `pairs` adds
    a surface for each listed pair of parents: the part of x that depends on
    both together (an interaction). A node without parents is just constant + noise.
    The curves are fitted as if the noise were a bell of one size; then its real size
    and shape are read off the leftovers (see noise_model.py).

    `scale` is plain by default; a log or log-odds scale writes the equation for
    log x or the log-odds of x instead (see scale.py).
    `fitting` decides how the constant, curves and surfaces are chosen (see
    fitting.py). `screening` (a PairScreening, see screening.py) chooses the pairs
    from the data; without it the node keeps `pairs` as given.

    `kinds` says how each parent is measured (see Background): a continuous
    parent gets a curve (Spline), a binary, ordinal or nominal one gets one
    height per level (Levels). A parent not listed is continuous.

    `noise_floor`: nothing is trusted to be exact. The variable and each term's push
    are trusted only up to this share of their own variance, so the noise variance is
    never below

        floor = share × (variance of x + Σ variance of each term's push)

    Without it an exact formula (z = x − y) scores without limit. And it is the same
    number whichever way a formula is written (x = z + y), so the rewriting does not
    decide the direction. 0, the default, adds nothing.
    """

    surface_intervals = 4  # a surface is coarser than a curve: it has far more heights

    def __init__(self, name: str, parents: list[str], pairs=(), intervals: int = 8,
                 fitting: BayesianFitting | None = None, screening: PairScreening | None = None, noise_floor: float = 0.0,
                 kinds: dict[str, str] | None = None, scale: Scale | None = None):
        self.name = name
        self.parents = list(parents)
        self.pairs: list[tuple[str, str]] = [(first, second) for first, second in pairs]
        self.intervals = intervals
        self.fitting = fitting if fitting is not None else BayesianFitting()
        self.screening = screening
        self.noise_floor = noise_floor
        self.kinds = dict(kinds) if kinds is not None else {}
        self.scale = scale if scale is not None else Scale()
        self._check_pairs()
        self.constant = 0.0
        self.curves: dict[str, Spline | Levels] = {}
        self.surfaces: dict[tuple[str, str], Surface] = {}
        self.smoothness: dict[str, float] = {}  # λ per smoothness block
        self.covariance = None                  # how unsure all stored heights are, together
        self.rows: dict[str, slice] = {}        # which rows of `covariance` belong to each term
        self.evidence = -np.inf                 # how probable the data was under the fit; set by fit
        self.noise_model = NoiseModel()         # the noise's size and shape; set by fit

    def fit(self, data: dict[str, np.ndarray]) -> None:
        """Choose the pairs (if a screening is set), then fit."""
        if self.screening is not None:
            self.screening.choose(self, data)
        else:
            self.refit(data)

    def refit(self, data: dict[str, np.ndarray]) -> None:
        """Fit with the current pairs: place the knots, then let the fitting choose
        the constant, curves and surfaces."""
        self.place_knots(data)
        design = self.design(data)
        solve = self.solver(design, self.target(data))
        result = self.fitting.fit(solve, self.size(design), self.smoothness_blocks())
        self.store(result, data)
        self.read_noise(result, data)

    def read_noise(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """Read the noise's size and shape off the leftovers (see NoiseModel), and add to
        the evidence what they gain over a bell of one size."""
        total = np.broadcast_to(self.total(data), len(data[self.name]))
        self.evidence += self.noise_model.read(self.noise(data), total, fit.noise_variance)

    def total(self, values: dict):
        """constant + Σ curves + Σ surfaces."""
        total = self.constant
        for parent, curve in self.curves.items():
            total = total + curve(values[parent])
        for (first, second), surface in self.surfaces.items():
            total = total + surface(values[first], values[second])
        return total

    def predict(self, values: dict):
        """What the parents explain, in the variable's own units."""
        return self.scale.backward(self.total(values))

    def noise(self, values: dict):
        """What the parents do not explain: scale(observed value) minus the sum."""
        return self.scale.forward(values[self.name]) - self.total(values)

    def leftovers(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """The noise as one column per number predicted (what a PairScreening looks at)."""
        return self.noise(data)[:, None]

    def terms(self) -> dict[str, Spline | Levels | Surface]:
        """Every curve and surface, by label: a curve is named after its parent,
        a surface after its pair, like "income×savings"."""
        terms: dict[str, Spline | Levels | Surface] = dict(self.curves)
        for pair, surface in self.surfaces.items():
            terms[self.surface_label(pair)] = surface
        return terms

    @staticmethod
    def surface_label(pair: tuple[str, str]) -> str:
        return f"{pair[0]}×{pair[1]}"

    # The steps of a fit. They are public because ChangeFit (change.py) runs
    # them itself, on people from two windows at once.

    def is_continuous(self, parent: str) -> bool:
        return self.kinds.get(parent, "continuous") == "continuous"

    def target(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """The variable's values on its scale: what the sum of curves aims at."""
        return self.scale.forward(data[self.name])

    def place_knots(self, data: dict[str, np.ndarray]) -> None:
        """Spread each curve's knots evenly over the range its parent takes in the
        data (a categorical parent gets one height per level seen instead); each
        surface does the same in both directions, with fewer intervals."""
        self.curves = {parent: self._edge(parent, data[parent], self.intervals) for parent in self.parents}
        self.surfaces = {}
        for first, second in self.pairs:
            surface = Surface(self._edge(first, data[first], self.surface_intervals),
                              self._edge(second, data[second], self.surface_intervals))
            surface.centre_on(data[first], data[second])
            self.surfaces[(first, second)] = surface

    def _edge(self, parent: str, values: np.ndarray, intervals: int) -> Spline | Levels:
        if self.is_continuous(parent):
            return Spline(values.min(), values.max(), intervals)
        return Levels(values, ordered=self.kinds[parent] != "nominal")

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
        """One smoothness belief per curve (it bends little) and per surface (it is flat)."""
        columns = self.columns()
        blocks = [SmoothnessBlock(parent, columns[parent], self.curves[parent].bending())
                  for parent in self.parents]
        for pair in self.pairs:
            label = self.surface_label(pair)
            blocks.append(SmoothnessBlock(label, columns[label], self.surfaces[pair].flattening()))
        return blocks

    def size(self, design: np.ndarray) -> int:
        """How many numbers a fit chooses: one per design column here. Nodes
        with more numbers (cut-points, a score per category) say so after
        `solver` has seen the data."""
        return design.shape[1]

    def solver(self, design: np.ndarray, target: np.ndarray) -> Solve:
        """A solve function for a fitting: hand it a penalty, get back a Fit.

        The noise is taken as bell-shaped here, so everything has a formula. The coefficients come
        from penalised least squares, the noise variance (its average square) is
        estimated from what is left over, plus the noise floor (see Node), and

            evidence = −N/2 · log(noise variance) + ½ log|penalty| − ½ log|WᵀW + penalty|

        The first part rewards a good fit, the rest charges for freedom that was
        used. The target is centred first, so its level (a balance of 250,000, say)
        does not count as a large coefficient.
        """
        middle = float(np.mean(target))
        centred = target - middle
        gram, pull = design.T @ design, design.T @ centred
        terms = list(self.columns().values())

        def floor(coefficients: np.ndarray) -> float:
            if self.noise_floor == 0.0:
                return 0.0
            pushes = sum(np.var(design[:, term] @ coefficients[term]) for term in terms)
            return self.noise_floor * float(np.var(target) + pushes)

        def solve(penalty: np.ndarray) -> Fit:
            inverse = np.linalg.inv(gram + penalty)
            coefficients = inverse @ pull
            leftover = np.sum((centred - design @ coefficients) ** 2) + coefficients @ penalty @ coefficients
            variance = float(leftover / len(target)) + floor(coefficients)
            evidence = -0.5 * len(target) * np.log(variance) + 0.5 * log_det(penalty) - 0.5 * log_det(gram + penalty)
            coefficients[0] += middle
            return Fit(coefficients, variance * inverse, float(evidence), variance)

        return solve

    def store(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """Hand the fitted numbers to the constant, the curves and the surfaces.

        Bump weights add up to 1, so raising a curve and lowering the constant
        by the same amount gives the same predictions. To make the split unique,
        each curve is shifted to average zero over the data and the constant
        takes the difference. A surface's free numbers are turned into its grid.
        """
        self.constant = float(fit.coefficients[0])
        self.evidence = fit.evidence + float(np.sum(self.scale.stretch(data[self.name])))
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
