"""Which pairs of parents get a surface, decided from the data."""

from itertools import combinations

import numpy as np


class PairScreening:
    """Decides which pairs of a node's parents need a surface.

    1. Fit the node with curves only.
    2. Rank every pair of parents by how much of what the curves leave
       unexplained one constant per quadrant can explain, at the best of a few
       cut points (FAST, Lou et al. 2013). An interaction shows up as quadrants
       that the curves get wrong in opposite directions.
    3. For the best-ranked pairs in turn, refit with that surface added and
       keep it only if the evidence rises by more than `margin`: the data must
       become more than e^margin times more probable (e^3 ≈ 20).
    """

    def __init__(self, candidates: int = 3, margin: float = 3.0, cuts=(0.25, 0.5, 0.75)):
        self.candidates = candidates
        self.margin = margin
        self.cuts = cuts

    def choose(self, node, data: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        node.pairs = []
        node.refit(data)
        best = node.evidence
        leftover = data[node.name] - node.predict(data)
        chosen: list[tuple[str, str]] = []
        for pair in self.rank(node.parents, data, leftover)[:self.candidates]:
            node.pairs = chosen + [pair]
            node.refit(data)
            if node.evidence > best + self.margin:
                chosen, best = chosen + [pair], node.evidence
        return chosen

    def rank(self, parents: list[str], data: dict[str, np.ndarray], leftover: np.ndarray) -> list[tuple[str, str]]:
        """Every pair of parents, most promising first."""
        scores = {pair: self._quadrant_score(data[pair[0]], data[pair[1]], leftover)
                  for pair in combinations(parents, 2)}
        return sorted(scores, key=lambda p: scores[p], reverse=True)

    def _quadrant_score(self, x1: np.ndarray, x2: np.ndarray, leftover: np.ndarray) -> float:
        """The share of the leftovers that one constant per quadrant explains,
        at the best of the cut points tried."""
        best = 0.0
        for cut1 in np.quantile(x1, self.cuts):
            for cut2 in np.quantile(x2, self.cuts):
                explained = 0.0
                for quadrant in ((x1 <= cut1) & (x2 <= cut2), (x1 <= cut1) & (x2 > cut2),
                                 (x1 > cut1) & (x2 <= cut2), (x1 > cut1) & (x2 > cut2)):
                    if quadrant.any():
                        explained += leftover[quadrant].sum() ** 2 / quadrant.sum()
                best = max(best, explained)
        return best / np.sum(leftover ** 2)
