import numpy as np


class RandomFeatures:
    """transformFeatures.m: random Fourier features. The inner product of two
    points' features approximates exp(-|a - b|²/2), so dividing the data by a width
    first gives a Gaussian kernel of that width without forming the T × T matrix.
    1,000 features from 1,000 rows on, 500 below. Returned as features × rows, as
    in MATLAB."""

    def __init__(self, random: np.random.Generator):
        self.random = random

    def transform(self, x: np.ndarray) -> np.ndarray:
        rows, columns = x.shape
        count = 1000 if rows >= 1000 else 500
        omega = self.random.standard_normal((columns, count))
        beta = self.random.uniform(0, 2 * np.pi, count)
        return (np.cos(x @ omega + beta) * np.sqrt(2 / count)).T
