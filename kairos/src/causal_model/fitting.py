"""How a node chooses its constant and heights.

A node gives its fitting a `solve` function: hand it a penalty and it returns
the most believable coefficients under that penalty, as a `Fit`. The fitting
decides which penalty to use. A fitting keeps no state of its own, so one
object can be shared by every node.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

import numpy as np

# Smoothness strengths BayesianFitting tries: 0.1, 0.32, 1, 3.2, ..., 10000.
GRID = tuple(10.0 ** np.arange(-1.0, 4.5, 0.5))


@dataclass
class Fit:
    coefficients: np.ndarray  # the constant, then each parent's heights
    covariance: np.ndarray    # how unsure each coefficient is, and which ones move together
    evidence: float           # how probable the data was under this penalty (higher is better)
    smoothness: list[float] = field(default_factory=list)  # λ per parent, empty for a plain fit


Solve = Callable[[np.ndarray], Fit]


class Fitting(Protocol):
    """What every fitting must offer."""

    def fit(self, solve: Solve, size: int, edges: list[slice]) -> Fit:
        """size: how many coefficients. edges: which coefficients are each parent's heights."""
        ...


class PlainFitting:
    """Only the data decides the heights.

    The tiny ridge is a very weak belief that every coefficient is near zero.
    It is needed because a curve and the constant can trade level (bump weights
    add up to 1), so without it there is no single answer.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, solve: Solve, size: int, edges: list[slice]) -> Fit:
        return solve(self.ridge * np.eye(size))


class BayesianFitting:
    """Bayesian splines: a belief that curves bend little, with its own strength λ per edge.

    The belief costs each curve its second differences, h[k-1] - 2 h[k] + h[k+1],
    so straight lines cost nothing and sharp bends cost a lot. λ near 0 trusts
    only the data; a huge λ allows only straight lines.

    Each λ is picked from `grid` as the value under which the data was most
    probable (the evidence). Edges are tuned one at a time with the others held
    fixed, for up to `rounds` passes or until no change helps.
    """

    def __init__(self, grid=GRID, rounds: int = 3, ridge: float = 1e-3):
        self.grid = list(grid)
        self.rounds = rounds
        self.ridge = ridge

    def fit(self, solve: Solve, size: int, edges: list[slice]) -> Fit:
        smoothness = [self.grid[len(self.grid) // 2]] * len(edges)
        best = solve(self._penalty(smoothness, size, edges))
        for _ in range(self.rounds):
            improved = False
            for edge in range(len(edges)):
                for strength in self.grid:
                    if strength == smoothness[edge]:
                        continue
                    trial = smoothness.copy()
                    trial[edge] = strength
                    candidate = solve(self._penalty(trial, size, edges))
                    if candidate.evidence > best.evidence:
                        best, smoothness, improved = candidate, trial, True
            if not improved:
                break
        return replace(best, smoothness=smoothness)

    def _penalty(self, smoothness: list[float], size: int, edges: list[slice]) -> np.ndarray:
        penalty = self.ridge * np.eye(size)
        for strength, columns in zip(smoothness, edges):
            count = columns.stop - columns.start
            bends = np.diff(np.eye(count), 2, axis=0)  # one row per second difference
            penalty[columns, columns] += strength * bends.T @ bends
        return penalty
