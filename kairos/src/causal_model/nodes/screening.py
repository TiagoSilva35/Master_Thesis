"""Which pairs of parents get a surface, decided from the data."""

from itertools import combinations

import numpy as np


class PairScreening:
    """Decides which pairs of a node's parents need a surface.

    1. Fit the node with curves only.
    2. Rank every pair by a table: people are grouped by both parents (quarters of a
       continuous one, levels of a categorical one), and the leftovers of each group
       are averaged. Without an interaction the averages are near zero; with one, some
       groups are too high and others too low. Pairs whose table is no larger than
       chance allows are dropped.
    3. For the best-ranked pairs in turn, refit with that surface added and keep it
       only if the evidence rises by more than `margin` (the data must become more
       than e^margin times more probable; e^3 ≈ 20). Stop at the first one that does
       not earn it, or after `most` surfaces.
    """

    def __init__(self, most: int = 2, margin: float = 3.0):
        self.most = most
        self.margin = margin

    def choose(self, node, data: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        """The pairs to keep; leaves the node fitted with them."""
        node.pairs = []
        node.refit(data)
        chosen: list[tuple[str, str]] = []
        best = node.evidence
        for pair in self.rank(node, data)[:self.most]:
            node.pairs = chosen + [pair]
            node.refit(data)
            if node.evidence <= best + self.margin:
                node.pairs = chosen
                node.refit(data)
                break
            chosen, best = chosen + [pair], node.evidence
        return chosen

    def rank(self, node, data: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        """The pairs whose table of leftovers stands out, most promising first."""
        if len(node.parents) < 2:
            return []
        leftovers = node.leftovers(data)
        groups = {parent: self._groups(data[parent], node.is_continuous(parent)) for parent in node.parents}
        signs = {pair: self._stands_out(groups[pair[0]], groups[pair[1]], leftovers)
                 for pair in combinations(node.parents, 2) if min(np.ptp(groups[parent]) for parent in pair) > 0}
        return sorted((pair for pair in signs if signs[pair] > 2), key=lambda pair: signs[pair], reverse=True)

    @staticmethod
    def _groups(values: np.ndarray, continuous: bool) -> np.ndarray:
        if continuous:
            return np.searchsorted(np.quantile(values, [0.25, 0.5, 0.75]), values)
        return np.unique(values, return_inverse=True)[1]

    @staticmethod
    def _stands_out(first: np.ndarray, second: np.ndarray, leftovers: np.ndarray) -> float:
        """How many typical deviations the table's size lies above what chance gives:
        Σ (sum of a group's leftovers)² / its people, in units of the leftovers' variance,
        is about the number of free cells when nothing is there, ± √(2 · free cells)."""
        cell = first * (second.max() + 1) + second
        people = np.bincount(cell)
        used = np.flatnonzero(people)
        size = sum(np.sum(np.bincount(cell, column)[used] ** 2 / people[used]) / max(np.mean(column ** 2), 1e-300)
                   for column in leftovers.T)
        free_cells = max(first.max() * second.max(), 1) * leftovers.shape[1]
        return float((size - free_cells) / np.sqrt(2 * free_cells))
