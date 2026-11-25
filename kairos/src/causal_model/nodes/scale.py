"""The scale a variable's equation is written on."""

import numpy as np


class Scale:
    """Plain: x = sum of curves + noise.

    Some equations are a sum only after undoing a bend. x = e^(a + b) is not a sum
    of curves of a and b, but log x = a + b is. A scale is that undoing:

        forward(x) = constant + Σ curves + Σ surfaces + noise

    so x = backward(the sum + noise). Nothing is known beforehand about which scale a
    variable needs; the evidence chooses among those its values allow.

    Scores on different scales can be compared because of `stretch`. A log squeezes
    large values together, so the leftovers look smaller even when the fit is no
    better. Each person's density is multiplied by how much the scale stretches the
    axis at their value (the change-of-variables rule), and `stretch` is its log.
    """

    name = "plain"

    def allows(self, values: np.ndarray) -> bool:
        return True

    def forward(self, values):
        return values

    def backward(self, total):
        return total

    def stretch(self, values: np.ndarray) -> np.ndarray:
        """log(slope of `forward`) at each value."""
        return np.zeros(len(values))


class LogScale(Scale):
    """log x, for positive values: products become sums."""

    name = "log"

    def allows(self, values: np.ndarray) -> bool:
        return bool(values.min() > 0)

    def forward(self, values):
        return np.log(values)

    def backward(self, total):
        return np.exp(total)

    def stretch(self, values: np.ndarray) -> np.ndarray:
        return -np.log(values)


class LogOddsScale(Scale):
    """log(x / (1 − x)), for values strictly between 0 and 1: a probability or share
    that follows an S-shaped curve becomes a sum."""

    name = "log-odds"

    def allows(self, values: np.ndarray) -> bool:
        return bool(values.min() > 0 and values.max() < 1)

    def forward(self, values):
        return np.log(values) - np.log1p(-np.asarray(values))

    def backward(self, total):
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(total, dtype=float)))

    def stretch(self, values: np.ndarray) -> np.ndarray:
        return -np.log(values) - np.log1p(-values)


SCALES: tuple[Scale, ...] = (Scale(), LogScale(), LogOddsScale())
