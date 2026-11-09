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
    coefficients: np.ndarray  # the constant, then each term's numbers (curves, then surfaces)
    covariance: np.ndarray    # how unsure each coefficient is, and which ones move together
    evidence: float           # how probable the data was under this penalty (higher is better)
    smoothness: list[float] = field(default_factory=list)  # λ per smoothness block, empty for a plain fit


@dataclass
class SmoothnessBlock:
    """One smoothness belief with its own λ: which coefficients it covers and
    how bending is measured there (a curve has one block, a surface two)."""

    label: str
    columns: slice
    bending: np.ndarray
    start: float | None = None  # λ the search starts from (None: the middle of the grid)


Solve = Callable[[np.ndarray], Fit]


class Fitting(Protocol):
    """What every fitting must offer."""

    def fit(self, solve: Solve, size: int, blocks: list[SmoothnessBlock]) -> Fit:
        """size: how many coefficients. blocks: one smoothness belief per λ."""
        ...


class PlainFitting:
    """Only the data decides the heights.

    The tiny ridge is a very weak belief that every coefficient is near zero.
    It is needed because a curve and the constant can trade level (bump weights
    add up to 1), so without it there is no single answer.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, solve: Solve, size: int, blocks: list[SmoothnessBlock]) -> Fit:
        return solve(self.ridge * np.eye(size))




class BayesianFitting:
    """Bayesian splines: a belief that curves and surfaces bend little, with its
    own strength λ per smoothness block (one per curve, two per surface).

    The belief costs bending (second differences, see Spline.bending), so
    straight lines cost nothing and sharp bends cost a lot. λ near 0 trusts
    only the data; a huge λ allows only straight lines.

    Each λ is picked from `grid` as the value under which the data was most
    probable (the evidence). Blocks are tuned one at a time with the others
    held fixed, for up to `rounds` passes or until no change helps.
    """

    def __init__(self, grid=GRID, rounds: int = 3, ridge: float = 1e-3):
        self.grid = list(grid)
        self.rounds = rounds
        self.ridge = ridge

    def fit(self, solve: Solve, size: int, blocks: list[SmoothnessBlock]) -> Fit:
        middle = self.grid[len(self.grid) // 2]
        smoothness = [block.start if block.start is not None else middle for block in blocks]
        best = solve(self.penalty(smoothness, size, blocks))
        for _ in range(self.rounds):
            improved = False
            for block in range(len(blocks)):
                for strength in self.grid:
                    if strength == smoothness[block]:
                        continue
                    trial = smoothness.copy()
                    trial[block] = strength
                    candidate = solve(self.penalty(trial, size, blocks))
                    if candidate.evidence > best.evidence:
                        best, smoothness, improved = candidate, trial, True
            if not improved:
                break
        return replace(best, smoothness=smoothness)

    def penalty(self, smoothness: list[float], size: int, blocks: list[SmoothnessBlock]) -> np.ndarray:
        """The full penalty matrix for these λ values: the tiny ridge plus every block."""
        penalty = self.ridge * np.eye(size)
        for strength, block in zip(smoothness, blocks):
            penalty[block.columns, block.columns] += strength * block.bending
        return penalty
