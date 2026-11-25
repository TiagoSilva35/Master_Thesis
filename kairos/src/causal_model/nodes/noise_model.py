"""What a continuous variable's noise looks like: its size and its shape."""

import numpy as np

from src.causal_model.nodes.noise_shape import BELL, SHAPES, NoiseShape

GROWTHS = (0.0, -1.2, -0.8, -0.4, 0.4, 0.8, 1.2)


class NoiseModel:
    """Reads the noise off the leftovers of a fitted node.

    - Size. The noise may grow or shrink with the prediction (large incomes vary more
      than small ones; a share near 0 or 1 varies less than one near a half):

          size² = level · e^(growth · position)

      where position is how far the prediction is from its average, in spreads. One
      number, growth, picked from a few values; 0 is one size for everyone.
    - Shape. The leftovers divided by their size have one of 15 shapes (NoiseShape).

    Both are picked together: a lopsided noise can look like a growing one, and only
    the pair tells them apart. `read` returns how much more probable the leftovers are
    than under a bell of one size, after paying for the choices: ½ log N for a growth,
    log 15 for a shape.

    This is what tells directions apart. Under the true graph the leftovers have one
    shape, unrelated to the causes; under a wrong one they usually do not, and no
    choice here describes them well.
    """

    def __init__(self):
        self.growth = 0.0
        self.shape: NoiseShape = BELL

    def read(self, leftovers: np.ndarray, prediction: np.ndarray, variance: float) -> float:
        """`variance`: the fitted noise variance, floor included. Where the floor is most
        of it there is little noise to have a shape, so the shape's gain shrinks with it."""
        floor = max(variance - float(np.mean(leftovers ** 2)), 0.0)
        position = np.clip((prediction - prediction.mean()) / max(prediction.std(), 1e-300), -2.5, 2.5)
        plain = self._fit(leftovers, np.full(len(leftovers), variance))
        self.growth, self.shape, best = 0.0, BELL, 0.0
        for growth in GROWTHS if prediction.std() > 0 else GROWTHS[:1]:
            size = np.full(len(leftovers), variance) if growth == 0 else self._size(leftovers, position, growth, floor)
            price = 0.5 * np.log(len(leftovers)) if growth else 0.0
            growing = self._fit(leftovers, size) - plain - price
            for shape in SHAPES:
                shaped = 0.0 if shape is BELL else ((1 - floor / variance) * shape.gain(leftovers / np.sqrt(size))
                                                    - np.log(len(SHAPES)))
                if growing + shaped > best:
                    self.growth, self.shape, best = growth, shape, growing + shaped
        return best

    @staticmethod
    def _size(leftovers: np.ndarray, position: np.ndarray, growth: float, floor: float) -> np.ndarray:
        """Each person's size², with the level that fits the leftovers best."""
        curve = np.exp(growth * position)
        return np.mean(leftovers ** 2 / curve) * curve + floor

    @staticmethod
    def _fit(leftovers: np.ndarray, size: np.ndarray) -> float:
        """The log-likelihood of the leftovers under bells of these sizes²."""
        return float(np.sum(-0.5 * np.log(2 * np.pi * size) - leftovers ** 2 / (2 * size)))
