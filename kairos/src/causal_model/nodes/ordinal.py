"""A variable with levels in a known order."""

import numpy as np

from src.causal_model.nodes.fitting import Fit, Solve
from src.causal_model.nodes.laplace import LaplaceNode


class OrdinalNode(LaplaceNode):
    """A variable whose levels have an order but no distances, like a credit
    band 0-4: the ordered logit. One score

        s = constant + Σ curve_parent(x_parent) + Σ surfaces

    pushes a person up the levels, and cut-points γ mark where one level ends:

        P(x ≤ level c) = σ(γ_c − s),   γ_0 = 0 < γ_1 < ... < γ_(k−2)

    γ_0 is fixed at 0 because the constant already sets the overall level. A
    fit's numbers are the design's coefficients followed by γ_1, ..., γ_(k−2).

    Levels that never occur in the data are dropped, since nothing can be learnt
    about them: `levels` keeps the values seen, in order, and level c means
    levels[c].
    """

    levels = np.zeros(0)      # the values seen, in increasing order; set when the node is fitted
    cut_points = np.zeros(0)  # γ_0 = 0, γ_1, ..., γ_(k−2); set when the node is fitted

    def solver(self, design: np.ndarray, target: np.ndarray) -> Solve:
        self.levels, codes = np.unique(target, return_inverse=True)
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: an ordered variable needs at least two levels in the data")
        return super().solver(design, codes.astype(float))

    def size(self, design: np.ndarray) -> int:
        return design.shape[1] + len(self.levels) - 2

    def store(self, fit: Fit, data: dict[str, np.ndarray]) -> None:
        free = len(self.levels) - 2
        self.cut_points = np.concatenate([[0.0], fit.coefficients[len(fit.coefficients) - free:]])
        super().store(fit, data)

    def probabilities(self, values: dict) -> np.ndarray:
        """P(x = levels[c]) for each person (rows) and level (columns)."""
        score = np.atleast_1d(self.total(values))
        edges = np.concatenate([[-np.inf], self.cut_points, [np.inf]])
        below = self.sigmoid(edges[None, :] - score[:, None])
        return np.diff(below, axis=1)

    def predict(self, values: dict):
        """The expected level, in the original values."""
        return self.probabilities(values) @ self.levels

    # --- the likelihood -------------------------------------------------------

    def _start_point(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        """The cut-points that reproduce each level's share of the data, with
        flat curves: P(x ≤ c) = share at or below c."""
        shares = np.cumsum(np.bincount(target.astype(int), minlength=len(self.levels)))[:-1] / len(target)
        log_odds = np.log(shares / (1 - shares))
        numbers = np.zeros(self.size(design))
        numbers[0] = -log_odds[0]
        numbers[design.shape[1]:] = log_odds[1:] - log_odds[0]
        return numbers

    def _edges(self, design: np.ndarray, target: np.ndarray, numbers: np.ndarray):
        """For each person: a = γ_c − s and b = γ_(c−1) − s around their level c."""
        edges = np.concatenate([[-np.inf, 0.0], numbers[design.shape[1]:], [np.inf]])
        score = design @ numbers[:design.shape[1]]
        codes = target.astype(int)
        return edges[codes + 1] - score, edges[codes] - score

    def _log_likelihood(self, design: np.ndarray, target: np.ndarray, numbers: np.ndarray) -> float:
        """log(σ(a) − σ(b)) = log σ(a) + log σ(−b) + log(1 − e^(b−a)), which stays
        accurate far in the tails. Cut-points out of order give −∞."""
        upper, lower = self._edges(design, target, numbers)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(upper > lower, np.log1p(-np.exp(lower - upper)), -np.inf)
        return float(np.sum(-np.logaddexp(0.0, -upper) - np.logaddexp(0.0, lower) + gap))

    def _slope_and_curvature(self, design: np.ndarray, target: np.ndarray,
                             numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """With p = σ(a) − σ(b), f = σ(1 − σ) and f' = f(1 − 2σ), each person's
        log p has slopes g_a = f(a)/p and g_b = −f(b)/p in a and b, and
        curvatures g_aa, g_bb, g_ab. The score s enters both a and b with sign −1;
        γ_c enters a, γ_(c−1) enters b."""
        upper, lower = self._edges(design, target, numbers)
        chance = np.maximum(self.sigmoid(upper) - self.sigmoid(lower), 1e-300)
        f_upper, f_lower = self._density(upper), self._density(lower)
        g_a, g_b = f_upper / chance, -f_lower / chance
        g_aa = self._density_slope(upper) / chance - g_a ** 2
        g_bb = -self._density_slope(lower) / chance - g_b ** 2
        g_ab = -g_a * g_b
        on_upper, on_lower = self._cut_point_columns(design, target)
        columns = design.shape[1]
        slope = np.concatenate([design.T @ (-(g_a + g_b)), on_upper.T @ g_a + on_lower.T @ g_b])
        hessian = np.zeros((len(numbers), len(numbers)))
        hessian[:columns, :columns] = design.T @ (design * (g_aa + 2 * g_ab + g_bb)[:, None])
        across = design.T @ (on_upper * -(g_aa + g_ab)[:, None] + on_lower * -(g_ab + g_bb)[:, None])
        hessian[:columns, columns:] = across
        hessian[columns:, :columns] = across.T
        between = on_upper.T @ (on_lower * g_ab[:, None])
        hessian[columns:, columns:] = (on_upper.T @ (on_upper * g_aa[:, None]) + on_lower.T @ (on_lower * g_bb[:, None])
                                       + between + between.T)
        return slope, -hessian

    def _cut_point_columns(self, design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Which free cut-point (γ_1 ... γ_(k−2)) sits in each person's a and b."""
        free = len(self.levels) - 2
        codes = target.astype(int)
        on_upper = np.zeros((len(codes), free))
        on_lower = np.zeros((len(codes), free))
        for cut in range(1, free + 1):
            on_upper[codes == cut, cut - 1] = 1.0
            on_lower[codes - 1 == cut, cut - 1] = 1.0
        return on_upper, on_lower

    def _density(self, edge: np.ndarray) -> np.ndarray:
        """f = σ(1 − σ), the slope of σ; 0 at ±∞."""
        chance = self.sigmoid(edge)
        return chance * (1 - chance)

    def _density_slope(self, edge: np.ndarray) -> np.ndarray:
        """f' = f(1 − 2σ); 0 at ±∞."""
        chance = self.sigmoid(edge)
        return chance * (1 - chance) * (1 - 2 * chance)
