import numpy as np


class Spline:
    # curve(x) = Σ weight_k(x) * height_k, with one cubic B-spline bump per height.
    # Bumps sit on the knots plus one extra centre beyond each end: intervals + 3 heights.

    def __init__(self, low: float, high: float, intervals: int):
        if high <= low:
            raise ValueError(f"need low < high, got low={low}, high={high}")
        self.knots = np.linspace(low, high, intervals + 1)
        self.gap = self.knots[1] - self.knots[0]
        self.centers = np.linspace(low - self.gap, high + self.gap, intervals + 3)
        self.heights = np.zeros(intervals + 3)
        self.covariance = None or np.zeros((intervals + 3, intervals + 3))  # how unsure the heights are; set when the node is fitted

    def weights(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.knots[0], self.knots[-1])
        u = np.abs(x[..., None] - self.centers) / self.gap  
        near = (4.0 - 6.0 * u**2 + 3.0 * u**3) / 6.0          
        far = (2.0 - u) ** 3 / 6.0                            
        return np.where(u < 1.0, near, np.where(u < 2.0, far, 0.0))

    def __call__(self, x):
        return self.weights(x) @ self.heights

    def bending(self) -> np.ndarray:
        """How much the curve bends, as a matrix: heights @ bending @ heights
        = Σ (h[k-1] - 2 h[k] + h[k+1])². Straight lines cost nothing."""
        second_differences = np.diff(np.eye(len(self.heights)), 2, axis=0)
        return second_differences.T @ second_differences

    def flattening(self) -> np.ndarray:
        """How far the curve is from flat, as a matrix: heights @ flattening @ heights
        = Σ (h[k+1] - h[k])². Only a flat curve, which changes no prediction, costs nothing."""
        first_differences = np.diff(np.eye(len(self.heights)), 1, axis=0)
        return first_differences.T @ first_differences

    def zero_average_basis(self, x) -> np.ndarray:
        """Columns spanning every set of heights whose curve averages zero over x.

        A curve's average over x is `average @ heights`, so these are the
        directions at right angles to `average` (the rows after the first in its
        singular value decomposition).
        """
        average = self.weights(x).mean(axis=0)
        _, _, directions = np.linalg.svd(average[None, :])
        return directions[1:].T

    def band(self, x, width: float = 1.96):
        """The curve ± width × its spread at x. 1.96 spreads hold about 95% of a bell."""
        if self.covariance is None:
            raise ValueError("this curve has no spread yet; fit its node first")
        weights = self.weights(x)
        variance = np.sum((weights @ self.covariance) * weights, axis=-1)
        spread = np.sqrt(np.maximum(variance, 0.0))
        middle = weights @ self.heights
        return middle - width * spread, middle + width * spread
