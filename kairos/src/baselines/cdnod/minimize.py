"""minimize.m (C. E. Rasmussen, 2001-2006; shipped with KCI-test and GPML): conjugate
gradients with a line search. Ported step by step, constants and all, because
the kernel widths CD-NOD learns depend on where it stops.

Copyright (C) 2001 - 2006 by Carl Edward Rasmussen; GPML licence: free to copy, use
and modify for research or education, keeping this notice."""

import cmath
from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], tuple[float, np.ndarray]]
FAILURES = (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError)


class ConjugateGradients:
    """Polak-Ribière conjugate gradients; each line search brackets with cubic
    extrapolation, then interpolates until the Wolfe-Powell conditions hold.
    length > 0 caps the line searches, length < 0 the function evaluations
    (CD-NOD always passes -350)."""

    INT = 0.1    # don't re-evaluate within 0.1 of the limit of the current bracket
    EXT = 3.0    # extrapolate at most 3 times the current step
    MAX = 20     # at most 20 function evaluations per line search
    RATIO = 10   # largest allowed slope ratio
    SIG = 0.1    # Wolfe-Powell: largest allowed ratio between new and old slopes
    RHO = SIG / 2

    def __init__(self, length: int = -350, reduction: float = 1.0):
        self.length = length
        self.reduction = reduction

    def minimize(self, objective: Objective, x: np.ndarray) -> np.ndarray:
        length, i, line_search_failed = self.length, 0, False
        f0, df0 = objective(x)
        i += length < 0
        s = -df0
        d0 = -s @ s
        x3 = self.reduction / (1 - d0)
        while i < abs(length):
            i += length > 0
            best_x, best_f, best_df = x, f0, df0
            budget = self.MAX if length > 0 else min(self.MAX, -length - i)
            # 1. extrapolate until the step is long enough
            while True:
                x2, f2, d2, f3, df3 = 0.0, f0, d0, f0, df0
                success = False
                while not success and budget > 0:
                    try:
                        budget -= 1
                        i += length < 0
                        f3, df3 = objective(x + x3 * s)
                        if not np.isfinite(f3) or not np.all(np.isfinite(df3)):
                            raise ValueError("not finite")
                        success = True
                    except FAILURES:
                        x3 = (x2 + x3) / 2  # bisect and try again
                if f3 < best_f:
                    best_x, best_f, best_df = x + x3 * s, f3, df3
                d3 = df3 @ s
                if d3 > self.SIG * d0 or f3 > f0 + x3 * self.RHO * d0 or budget == 0:
                    break
                x1, f1, d1 = x2, f2, d2
                x2, f2, d2 = x3, f3, d3
                x3 = self._extrapolate(x1, f1, d1, x2, f2, d2)
            # 2. interpolate inside the bracket until the step is acceptable
            x4 = f4 = d4 = np.nan
            while (abs(d3) > -self.SIG * d0 or f3 > f0 + x3 * self.RHO * d0) and budget > 0:
                if d3 > 0 or f3 > f0 + x3 * self.RHO * d0:
                    x4, f4, d4 = x3, f3, d3
                else:
                    x2, f2, d2 = x3, f3, d3
                x3 = self._interpolate(x2, f2, d2, x4, f4, d4, f0)
                f3, df3 = objective(x + x3 * s)
                if f3 < best_f:
                    best_x, best_f, best_df = x + x3 * s, f3, df3
                budget -= 1
                i += length < 0
                d3 = df3 @ s
            # 3. accept the step and pick the next direction, or fall back
            if abs(d3) < -self.SIG * d0 and f3 < f0 + x3 * self.RHO * d0:
                x, f0 = x + x3 * s, f3
                s = (df3 @ df3 - df0 @ df3) / (df0 @ df0) * s - df3  # Polak-Ribière
                df0 = df3
                d3, d0 = d0, df0 @ s
                if d0 > 0:  # the new slope must point downhill
                    s = -df0
                    d0 = -s @ s
                x3 = x3 * min(self.RATIO, d3 / (d0 - np.finfo(float).tiny))
                line_search_failed = False
            else:
                x, f0, df0 = best_x, best_f, best_df
                if line_search_failed or i > abs(length):
                    break
                s = -df0
                d0 = -s @ s
                x3 = 1 / (1 - d0)
                line_search_failed = True
        return x

    def _extrapolate(self, x1, f1, d1, x2, f2, d2) -> float:
        """Cubic extrapolation, kept within [x2 + INT·(x2 - x1), EXT·x2]."""
        a = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
        b = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
        try:
            x3 = x1 - d1 * (x2 - x1) ** 2 / (b + cmath.sqrt(b * b - a * d1 * (x2 - x1)))
        except ZeroDivisionError:
            x3 = complex(np.inf)
        if x3.imag != 0 or not np.isfinite(x3.real) or x3.real < 0 or x3.real > x2 * self.EXT:
            return x2 * self.EXT
        return max(x3.real, x2 + self.INT * (x2 - x1))

    def _interpolate(self, x2, f2, d2, x4, f4, d4, f0) -> float:
        """Quadratic interpolation if f4 rose above f0, cubic otherwise; bisect on a
        numerical problem; never closer than INT to either end of the bracket.
        One difference: a complex cubic root, which MATLAB would carry on with
        (comparing complex numbers by size), bisects here like any other numerical
        problem."""
        try:
            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                a = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                b = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (cmath.sqrt(b * b - a * d2 * (x4 - x2) ** 2) - b) / a
                x3 = x3.real if x3.imag == 0 else complex(x3)
        except ZeroDivisionError:
            x3 = np.nan
        if isinstance(x3, complex) or not np.isfinite(x3):
            x3 = (x2 + x4) / 2
        return max(min(x3, x4 - self.INT * (x4 - x2)), x2 + self.INT * (x4 - x2))
