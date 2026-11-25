"""The shape of a continuous variable's noise."""

from math import exp, gamma, log, sqrt

import numpy as np


class NoiseShape:
    """One shape, always with average 0 and spread 1:

        density(z) = constant · e^(−(|position| / width)^power)

    - `power` says how peaked it is: 2 is the bell, 1 is peaked with long tails, 4 is
      flat with short tails.
    - `lean` makes the two sides different: the right side is e^lean times as wide as
      the middle, the left side e^−lean. 0 is symmetric.

    Why shapes matter: they tell directions apart. With bell-shaped noise a straight
    arrow fits equally well both ways. With any other shape only the true direction
    leaves noise of one shape, unrelated to the cause (the idea of LiNGAM, Shimizu et
    al. 2006).
    """

    def __init__(self, power: float, lean: float = 0.0):
        self.power = power
        self.lean = lean
        self.right, self.left = exp(lean), exp(-lean)
        middle = (self.right - self.left) * gamma(2 / power) / gamma(1 / power)  # its average if it peaked at 0
        square = gamma(3 / power) / gamma(1 / power) * (self.right**3 + self.left**3) / (self.right + self.left)
        self.average, self.spread = middle, sqrt(square - middle**2)
        self.constant = log(power * self.spread / (gamma(1 / power) * (self.right + self.left)))

    def log_density(self, z: np.ndarray) -> np.ndarray:
        position = z * self.spread + self.average  # distance from the peak
        width = np.where(position > 0, self.right, self.left)
        return self.constant - (np.abs(position) / width) ** self.power

    def gain(self, leftovers: np.ndarray) -> float:
        """How much more probable the leftovers are under this shape than under a bell
        of the same spread (in log units)."""
        z = (leftovers - leftovers.mean()) / max(leftovers.std(), 1e-300)
        bell = -0.5 * len(z) * log(2 * np.pi) - 0.5 * np.sum(z**2)
        return float(np.sum(self.log_density(z)) - bell)


    @property
    def name(self) -> str:
        peak = {1.0: "peaked", 2.0: "bell", 4.0: "flat"}.get(self.power, f"power {self.power:g}")
        side = "" if self.lean == 0 else (" leaning right" if self.lean > 0 else " leaning left")
        return peak + side


BELL = NoiseShape(2.0)
SHAPES = (BELL,) + tuple(NoiseShape(power, lean) for power in (1.0, 2.0, 4.0) for lean in (-1.2, -0.5, 0.0, 0.5, 1.2)
                         if (power, lean) != (2.0, 0.0))
