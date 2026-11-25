"""How a node chooses its constant and heights.

A node gives its fitting a `solve` function: hand it a penalty and it returns
the most believable coefficients under that penalty, as a `Fit`. The fitting
decides which penalty to use. It keeps no state of its own, so one object can
be shared by every node.
"""

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


@dataclass
class Fit:
    coefficients: np.ndarray  # the constant, then each term's numbers (curves, then surfaces)
    covariance: np.ndarray    # how unsure each coefficient is, and which ones move together
    evidence: float           # how probable the data was under this penalty (higher is better)
    noise_variance: float = 1.0  # σ̂² of a bell-shaped node; 1 for the others (their beliefs are not in units of noise)
    smoothness: list[float] = field(default_factory=list)  # λ per smoothness block


@dataclass
class SmoothnessBlock:
    """One smoothness belief with its own λ: which coefficients it covers and
    how bending is measured there."""

    label: str
    columns: slice
    bending: np.ndarray
    start: float | None = None  # λ the search starts from (None: the fitting's usual start)


Solve = Callable[[np.ndarray], Fit]


def log_det(matrix: np.ndarray) -> float:
    """log |matrix|, for the symmetric positive matrices used here."""
    return 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(matrix)))))


class BayesianFitting:
    """Bayesian splines: a belief that curves bend little and surfaces are flat,
    with its own strength λ per smoothness block.

    λ near 0 trusts only the data; a huge λ allows only straight lines (or, for a
    surface, nothing at all). Each λ is the one under which the data was most
    probable (the evidence). At that λ

        λ = σ² · freedom / bending

    where `freedom` is how many directions of the curve the data decided, not the
    belief (between 0 and its number of heights), and `bending` is how much the
    fitted curve bends (MacKay 1992; Wood & Fasiolo 2017). So the search repeats:
    fit, read off freedom and bending, set λ, until the evidence stops rising. A
    step that would lower the evidence is shortened.

    `ridge` is a very weak belief that every number is near zero. It is needed
    because a curve and the constant can trade level (bump weights add up to 1), so
    without it there is no single answer.
    """

    def __init__(self, ridge: float = 1e-3, start: float = 10.0, lowest: float = 1e-5, highest: float = 1e6,
                 rounds: int = 40):
        self.ridge = ridge
        self.start = start
        self.lowest = lowest
        self.highest = highest
        self.rounds = rounds

    def fit(self, solve: Solve, size: int, blocks: list[SmoothnessBlock]) -> Fit:
        """size: how many coefficients. blocks: one smoothness belief per λ."""
        strengths = np.array([block.start if block.start is not None else self.start for block in blocks])
        fit = solve(self.penalty(strengths, size, blocks))
        for _ in range(self.rounds if blocks else 0):
            trial = self._next(strengths, fit, blocks)
            candidate = solve(self.penalty(trial, size, blocks))
            while candidate.evidence < fit.evidence and np.abs(np.log(trial / strengths)).max() > 0.01:
                trial = np.sqrt(trial * strengths)  # half the step, on the log scale
                candidate = solve(self.penalty(trial, size, blocks))
            gain = candidate.evidence - fit.evidence
            if gain > 0:
                strengths, fit = trial, candidate
            if gain < 1e-3:
                break
        return replace(fit, smoothness=list(strengths))

    def penalty(self, strengths, size: int, blocks: list[SmoothnessBlock]) -> np.ndarray:
        """The full penalty matrix for these λ values: the tiny ridge plus every block."""
        penalty = self.ridge * np.eye(size)
        for strength, block in zip(strengths, blocks):
            penalty[block.columns, block.columns] += strength * block.bending
        return penalty

    def _next(self, strengths: np.ndarray, fit: Fit, blocks: list[SmoothnessBlock]) -> np.ndarray:
        """σ² · freedom / bending for each block, with freedom = λ · (trace before
        seeing the data − trace after). The power 1.5 takes longer steps."""
        new = strengths.copy()
        for index, (strength, block) in enumerate(zip(strengths, blocks)):
            numbers = fit.coefficients[block.columns]
            stiffness = np.maximum(np.linalg.eigvalsh(block.bending), 0.0)
            before = np.sum(stiffness / (self.ridge + strength * stiffness))
            after = np.sum(fit.covariance[block.columns, block.columns] * block.bending) / fit.noise_variance
            bending = numbers @ block.bending @ numbers / fit.noise_variance
            ratio = max(before - after, 1e-12) / max(bending, 1e-12)
            new[index] = np.clip(strength * np.clip(ratio ** 1.5, 0.01, 100.0), self.lowest, self.highest)
        return new
