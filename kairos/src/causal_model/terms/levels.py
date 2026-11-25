import numpy as np


class Levels:
    """The edge from a categorical parent: one height per level, instead of a
    curve over made-up numbers.

        term(x) = height of x's level

    The penalty pulls the heights together, with a strength λ chosen by the
    evidence like a curve's:

    - ordered (an ordinal or yes/no parent): differences between neighbouring
      levels, Σ (h[k+1] − h[k])² (ordinal smoothing, Gertheiss & Tutz 2009). Only
      the order is used, never the gaps between codes.
    - not ordered (a nominal parent): each height's distance from their average,
      Σ (h[k] − average)². There are no neighbours, so re-numbering the levels
      changes nothing.

    A very strong pull puts every level at the same height, so the parent has no
    effect: the same as leaving the arrow out. In between, a level with few rows
    borrows from the others.

    It offers what a Spline offers (weights, bending, flattening,
    zero_average_basis, band, outside, calling it), so a Node uses it the same way.
    A level never seen when the term was made counts as the average of all heights.
    """

    def __init__(self, values: np.ndarray, ordered: bool):
        self.levels = np.unique(np.asarray(values, dtype=float))
        self.ordered = ordered
        self.heights = np.zeros(len(self.levels))
        self.covariance: np.ndarray | None = None  # how unsure the heights are; set when the node is fitted

    def weights(self, x) -> np.ndarray:
        """1 for the person's level, 0 for the others; 1/k for every level when
        the value was never seen, so the term gives the average height."""
        x = np.asarray(x, dtype=float)
        weights = (x[..., None] == self.levels).astype(float)
        unseen = ~weights.any(axis=-1)
        if np.any(unseen):
            weights[unseen] = 1.0 / len(self.levels)
        return weights

    def outside(self, x) -> np.ndarray:
        """True where x is not one of the levels seen."""
        x = np.asarray(x, dtype=float)
        return ~np.isin(x, self.levels)

    def __call__(self, x):
        return self.weights(x) @ self.heights

    def bending(self) -> np.ndarray:
        """The smoothness cost as a matrix: heights @ bending @ heights is
        Σ (h[k+1] − h[k])² when ordered, Σ (h[k] − average)² when not."""
        count = len(self.levels)
        if self.ordered:
            differences = np.diff(np.eye(count), 1, axis=0)
            return differences.T @ differences
        return np.eye(count) - np.ones((count, count)) / count

    def flattening(self) -> np.ndarray:
        """How far the term is from flat: the same cost as `bending`, since for
        levels only a flat term costs nothing either way."""
        return self.bending()

    def zero_average_basis(self, x) -> np.ndarray:
        """Columns spanning every set of heights whose term averages zero over x
        (as Spline.zero_average_basis)."""
        average = self.weights(x).mean(axis=0)
        _, _, directions = np.linalg.svd(average[None, :])
        return directions[1:].T

    def band(self, x, width: float = 1.96):
        """The term ± width × its spread at x."""
        if self.covariance is None:
            raise ValueError("this term has no spread yet; fit its node first")
        weights = self.weights(x)
        variance = np.sum((weights @ self.covariance) * weights, axis=-1)
        spread = np.sqrt(np.maximum(variance, 0.0))
        middle = weights @ self.heights
        return middle - width * spread, middle + width * spread
