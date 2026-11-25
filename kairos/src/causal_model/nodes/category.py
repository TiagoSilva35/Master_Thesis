"""A variable with levels in no order."""

import copy

import numpy as np

from src.causal_model.nodes.fitting import Fit, SmoothnessBlock, Solve
from src.causal_model.nodes.laplace import LaplaceNode
from src.causal_model.nodes.node import Node


class CategoryNode(LaplaceNode):
    """A variable whose levels have no order, like an event type: the softmax.
    Every category except a reference has its own score

        s_c = constant_c + Σ curve_c,parent(x_parent) + Σ surfaces_c

    the reference has score 0, and

        P(x = c) = e^(s_c) / Σ_d e^(s_d)

    The reference is the most common category. Re-numbering the categories
    therefore changes nothing: the same category stays the reference, and the
    others are treated alike. Each curve has one λ, shared by every category.
    Categories that never occur in the data are dropped; a category asked about
    later but never seen gets the average score of those seen.

    A fit's numbers are laid out term by term: the constants of every category,
    then the first parent's heights for every category, and so on. Each curve's
    smoothness block then covers one stretch of numbers. After a fit, `sides`
    holds one plain Node per non-reference category, whose curves are that
    category's log-odds against the reference.
    """

    levels = np.zeros(0)   # the values seen, in increasing order; set when the node is fitted
    reference = 0.0        # the most common value; set when the node is fitted
    others = np.zeros(0)   # the other values seen, in increasing order

    sides: dict[float, Node] = {}          # one per non-reference category; set when the node is fitted
    _layout = np.zeros((0, 0), dtype=int)  # position of (category, design column) among the numbers

    def solver(self, design: np.ndarray, target: np.ndarray) -> Solve:
        self.levels, codes, counts = np.unique(target, return_inverse=True, return_counts=True)
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: a category variable needs at least two levels in the data")
        self.reference = float(self.levels[np.argmax(counts)])
        self.others = self.levels[self.levels != self.reference]
        self._layout = self._make_layout()
        which = np.searchsorted(self.others, target)  # row of each person's category in `others`
        which[target == self.reference] = -1
        return super().solver(design, which.astype(float))

    def size(self, design: np.ndarray) -> int:
        return len(self.others) * design.shape[1]

    def smoothness_blocks(self) -> list[SmoothnessBlock]:
        """The plain node's blocks, each stretched over every category: one λ per
        curve, shared."""
        blocks = []
        for block in super().smoothness_blocks():
            first, last = self._layout[0, block.columns.start], self._layout[-1, block.columns.stop - 1]
            bending = np.kron(np.eye(len(self.others)), block.bending)
            blocks.append(SmoothnessBlock(block.label, slice(first, last + 1), bending, block.start))
        return blocks

    def store(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        """One side node per non-reference category, holding its share of the numbers."""
        self.evidence = fit.evidence
        self.smoothness = {block.label: strength for block, strength in zip(self.smoothness_blocks(), fit.smoothness)}
        self.sides = {}
        for row, level in enumerate(self.others):
            side = Node(f"{self.name}={level:g}", self.parents, self.pairs, self.intervals, kinds=self.kinds)
            side.curves, side.surfaces = copy.deepcopy(self.curves), copy.deepcopy(self.surfaces)
            positions = self._layout[row]
            side.store(Fit(fit.coefficients[positions], fit.covariance[np.ix_(positions, positions)],
                           fit.evidence, smoothness=fit.smoothness), {**data, side.name: data[self.name]})
            self.sides[float(level)] = side

    def scores(self, values: dict, levels=None) -> np.ndarray:
        """Each category's score for each person (rows) and level (columns); the
        reference's is 0. A level never seen gets the average score of those seen."""
        seen = {float(level): np.atleast_1d(side.predict(values)) for level, side in self.sides.items()}
        reference = np.zeros_like(next(iter(seen.values())))
        seen[self.reference] = reference
        average = sum(seen.values(), reference) / len(seen)
        wanted = self.levels if levels is None else levels
        return np.column_stack([seen.get(float(level), average) for level in wanted])

    def probabilities(self, values: dict, levels=None) -> np.ndarray:
        """P(x = level) for each person (rows) and level (columns)."""
        scores = self.scores(values, levels)
        scores = scores - scores.max(axis=1, keepdims=True)
        return np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    def predict(self, values: dict):
        """The chance of each category seen, for each person (rows)."""
        return self.probabilities(values)

    def leftovers(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """One column per category: 1 or 0 for whether it was drawn, minus its chance."""
        return (data[self.name][:, None] == self.levels) - self.predict(data)

    # --- the likelihood -------------------------------------------------------

    def _make_layout(self) -> np.ndarray:
        """Term by term (constant, each curve, each surface), every category's
        columns side by side."""
        terms = [slice(0, 1)] + list(self.columns().values())
        width = 1 + sum(term.stop - term.start for term in terms[1:])
        layout = np.zeros((len(self.others), width), dtype=int)
        position = 0
        for term in terms:
            size = term.stop - term.start
            for row in range(len(self.others)):
                layout[row, term] = np.arange(position, position + size)
                position += size
        return layout

    def _start_point(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Flat curves, and constants that reproduce each category's share."""
        numbers = np.zeros(self.size(design))
        reference = np.sum(target == -1)
        for row in range(len(self.others)):
            numbers[self._layout[row, 0]] = np.log(np.sum(target == row) / reference)
        return numbers

    def _all_scores(self, design: np.ndarray, numbers: np.ndarray) -> np.ndarray:
        """Each person's score for the reference (0, first column) and every other category."""
        return np.hstack([np.zeros((len(design), 1)), design @ numbers[self._layout].T])

    def _log_likelihood(self, design: np.ndarray, target: np.ndarray, numbers: np.ndarray) -> float:
        scores = self._all_scores(design, numbers)
        chosen = scores[np.arange(len(target)), target.astype(int) + 1]
        return float(np.sum(chosen - np.logaddexp.reduce(scores, axis=1)))

    def _slope_and_curvature(self, design: np.ndarray, target: np.ndarray,
                             numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slope (y_c − π_c) · design per category; curvature
        designᵀ · π_c (δ_cd − π_d) · design per pair of categories."""
        scores = self._all_scores(design, numbers)
        chances = np.exp(scores - np.logaddexp.reduce(scores, axis=1, keepdims=True))[:, 1:]
        chosen = np.zeros_like(chances)
        rows = target >= 0
        chosen[np.arange(len(target))[rows], target[rows].astype(int)] = 1.0
        slope = np.zeros(len(numbers))
        slope[self._layout] = (chosen - chances).T @ design
        curvature = np.zeros((len(numbers), len(numbers)))
        for row in range(len(self.others)):
            for other in range(len(self.others)):
                weight = chances[:, row] * ((row == other) - chances[:, other])
                curvature[np.ix_(self._layout[row], self._layout[other])] = design.T @ (design * weight[:, None])
        return slope, curvature
