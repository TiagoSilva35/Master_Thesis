"""A pair edge: the part of a node that depends on two parents together."""

import numpy as np

from src.causal_model.terms.levels import Levels
from src.causal_model.terms.spline import Spline


class Surface:
    """A grid of heights over two parents, built from their two edges:

        surface(x1, x2) = Σ_j Σ_k weight_j(x1) · weight_k(x2) · height_jk

    A side is a Spline for a continuous parent and Levels for a categorical one:

        continuous × continuous    a smooth map (a tensor-product spline)
        categorical × continuous   one curve per level: how a gate is written
        categorical × categorical  one height per combination of levels

    The surface holds only the interaction: anything that depends on one parent
    alone belongs in that parent's curve. `centre_on` enforces this by allowing
    only grids whose every slice averages zero over the data (like the ti()
    terms of R's mgcv package). The grids that are still allowed are described
    by fewer numbers, the free numbers; `free` turns them into the full grid.
    """

    def __init__(self, first: Spline | Levels, second: Spline | Levels):
        self.first = first
        self.second = second
        self.heights = np.zeros(len(first.heights) * len(second.heights))  # the grid, row by row
        self.covariance: np.ndarray | None = None  # how unsure the grid is; set when the node is fitted
        self.free = np.eye(len(self.heights))

    def centre_on(self, x1: np.ndarray, x2: np.ndarray) -> None:
        """Allow only grids whose slices average zero over these data."""
        self.free = np.kron(self.first.zero_average_basis(x1), self.second.zero_average_basis(x2))

    def weights(self, x1, x2):
        """One weight per grid height: weight_j(x1) · weight_k(x2), row by row."""
        w1, w2 = self.first.weights(x1), self.second.weights(x2)
        return (w1[..., :, None] * w2[..., None, :]).reshape(*w1.shape[:-1], -1)

    def design(self, x1, x2):
        """The design columns for the free numbers."""
        return self.weights(x1, x2) @ self.free

    def flattening(self) -> np.ndarray:
        """The smoothness cost, for the free numbers: steps between neighbouring
        heights in both directions. Only a flat (zero) surface costs nothing, so a
        very strong pull switches the interaction off."""
        rows, columns = len(self.first.heights), len(self.second.heights)
        along_first = np.kron(self.first.flattening(), np.eye(columns))
        along_second = np.kron(np.eye(rows), self.second.flattening())
        return self.free.T @ (along_first + along_second) @ self.free

    def __call__(self, x1, x2):
        return self.weights(x1, x2) @ self.heights

    def band(self, x1, x2, width: float = 1.96):
        """The surface ± width × its spread at (x1, x2)."""
        if self.covariance is None:
            raise ValueError("this surface has no spread yet; fit its node first")
        weights = self.weights(x1, x2)
        variance = np.sum((weights @ self.covariance) * weights, axis=-1)
        spread = np.sqrt(np.maximum(variance, 0.0))
        middle = weights @ self.heights
        return middle - width * spread, middle + width * spread
