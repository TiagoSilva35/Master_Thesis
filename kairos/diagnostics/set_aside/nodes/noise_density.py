"""The shape of a set of numbers, learnt instead of assumed."""

import numpy as np

from src.causal_model.terms import Spline

# Smoothness strengths NoiseDensity tries: 0.1, 0.32, ..., 10^6. Higher than a
# curve's grid, because the density's limit (a bell) needs a very strong pull.
DENSITY_GRID = tuple(10.0 ** np.arange(-1.0, 6.5, 0.5))


class NoiseDensity:
    """Learns f, the shape of a set of numbers (a node's leftovers, say), as a
    smooth density:

        f(r) = e^s(r) / Z

    where s is a P-spline with `bumps` heights over the numbers' range, e^ keeps
    f positive, and Z makes its area 1 (density smoothing, Eilers & Marx 1996).

    How it is learnt:
    1. The numbers are sorted into about `bins` equal bins and counted.
    2. Bin j should hold about N × width × f(centre_j) numbers, so log(expected
       count_j) = s(centre_j) up to a constant, and the counts are fitted by
       Poisson regression. Its log-likelihood is concave in the heights, so
       Newton finds the single top.
    3. The penalty is on third differences of the heights, with λ chosen by the
       Laplace evidence of the counts. Third differences are zero for any
       parabola, and e^parabola is a bell: a very strong pull gives back the
       bell, and the data decide how far to move away from it.

    `smallest_width` keeps bins from being narrower than a scale that the
    numbers cannot resolve (the noise floor's, or a rounding step), so spikes
    from rounding or piled-up values earn nothing. Wider bins cover only the
    numbers' range (with at least `fewest_bins` bins), so the bumps are never
    spread over empty space and the shape keeps its detail.

    Each number is scored by the probability of its bin divided by the bin's
    width, the same quantity the Poisson fit estimates. With fine bins this is
    f at the number; for values on a grid (a rounded or discrete variable) it is
    the probability of the value, with bins centred on the values, instead of f
    read off between two bins, where a smooth curve over spikes has troughs.

    After `fit`: `log_density(r)` gives log f(r), and `occam` is what f's own
    freedom costs, −½ a·P·a + ½ log|P| − ½ log|H| (a the heights, H the curvature
    of the counts' log-likelihood plus P).
    """

    def __init__(self, bins: int = 100, bumps: int = 20, grid=DENSITY_GRID, ridge: float = 1e-3,
                 smallest_width: float = 0.0, steps: int = 50, fewest_bins: int = 10):
        self.bins = bins
        self.fewest_bins = fewest_bins
        self.bumps = bumps
        self.grid = list(grid)
        self.ridge = ridge
        self.smallest_width = smallest_width
        self.steps = steps
        self.spline: Spline | None = None
        self.edges = np.zeros(0)
        self.bin_log_density = np.zeros(0)  # log(probability of each bin / width)
        self.occam = 0.0
        self.smoothness = 0.0  # the λ chosen

    def fit(self, numbers: np.ndarray) -> None:
        edges = self._edges(numbers)
        counts = np.histogram(numbers, edges)[0].astype(float)
        self.spline = Spline(edges[0], edges[-1], self.bumps - 3)
        design = self.spline.weights((edges[:-1] + edges[1:]) / 2)
        third = np.diff(np.eye(self.bumps), 3, axis=0)
        bending = third.T @ third
        start = np.full(self.bumps, np.log(max(counts.mean(), 1e-12)))
        best = None
        for strength in self.grid:
            penalty = self.ridge * np.eye(self.bumps) + strength * bending
            heights = self._climb(design, counts, penalty, start)
            start = heights
            expected = np.exp(design @ heights)
            curvature = design.T @ (design * expected[:, None]) + penalty
            occam = (-0.5 * heights @ penalty @ heights + 0.5 * np.linalg.slogdet(penalty)[1]
                     - 0.5 * np.linalg.slogdet(curvature)[1])
            evidence = float(np.sum(counts * (design @ heights) - expected)) + occam
            if best is None or evidence > best[0]:
                best = (evidence, heights, occam, strength)
        assert best is not None
        _, self.spline.heights, self.occam, self.smoothness = best
        self.edges = edges
        self.bin_log_density = self._bin_log_density(edges)

    def log_density(self, numbers: np.ndarray) -> np.ndarray:
        """log(probability of the number's bin / width); numbers past the range
        count in the edge bins."""
        if self.spline is None:
            raise ValueError("fit the density first")
        width = self.edges[1] - self.edges[0]
        index = np.floor((np.asarray(numbers) - self.edges[0]) / width).astype(int)
        return self.bin_log_density[np.clip(index, 0, len(self.bin_log_density) - 1)]

    def _edges(self, numbers: np.ndarray) -> np.ndarray:
        """Equal bins at least `smallest_width` wide, the first centred on the
        smallest number (so values on a grid sit in the middle of their bins):
        about `bins` bins, fewer when they must be wider, never fewer than
        `fewest_bins` (extra bins are shared out at both ends)."""
        low, high = float(numbers.min()), float(numbers.max())
        width = max((high - low) / self.bins, self.smallest_width, 1e-12)
        needed = int(np.floor((high - low) / width + 0.5)) + 1
        count = max(self.fewest_bins, needed)
        start = low - width / 2 - (count - needed) // 2 * width
        return start + width * np.arange(count + 1)

    def _climb(self, design: np.ndarray, counts: np.ndarray, penalty: np.ndarray, start: np.ndarray) -> np.ndarray:
        """Newton's method on the Poisson log-likelihood minus the penalty, with
        each step halved until it climbs."""
        def height(heights):
            return float(np.sum(counts * (design @ heights) - np.exp(design @ heights)) - 0.5 * heights @ penalty @ heights)

        heights, current = start.copy(), height(start)
        for _ in range(self.steps):
            expected = np.exp(design @ heights)
            slope = design.T @ (counts - expected) - penalty @ heights
            step = np.linalg.solve(design.T @ (design * expected[:, None]) + penalty, slope)
            length = 1.0
            while True:
                trial = heights + length * step
                trial_height = height(trial)
                if trial_height >= current - 1e-9 or length < 1e-6:
                    break
                length /= 2
            heights, current = trial, trial_height
            if np.max(np.abs(length * step)) < 1e-8:
                break
        return heights

    def _bin_log_density(self, edges: np.ndarray) -> np.ndarray:
        """log(∫ over each bin of e^s / Z, divided by the width), with Z = ∫ e^s over
        all bins; the integrals by the trapezoid rule, 20 steps per bin."""
        assert self.spline is not None
        steps = 20
        points = np.linspace(edges[0], edges[-1], steps * (len(edges) - 1) + 1)
        values = self.spline(points)
        top = values.max()
        heights = np.exp(values - top)
        pieces = (heights[:-1] + heights[1:]) / 2 * np.diff(points)
        mass = pieces.reshape(len(edges) - 1, steps).sum(axis=1)
        width = edges[1] - edges[0]
        with np.errstate(divide="ignore"):
            return np.log(mass / mass.sum() / width)
