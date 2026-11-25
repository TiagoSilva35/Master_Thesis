from collections.abc import Callable

import numpy as np

from src.baselines.cdnod.meek import MeekRules

# (causes, effect, c_index) -> how dependent the two changes look; ChangeDependence in CD-NOD
Score = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


class IndependentChangeOrientation:
    """Phase 3 of nonsta_cd_new.m: orient links between changing variables (those
    with C → x) that phase 2 left undirected.

    Repeatedly, each remaining changing variable with an undirected link is scored
    as if it were the effect of all its undirected neighbours and its parents; the
    one with the smallest score (the most independent changes) is taken as the
    sink, its undirected links all point into it, and it leaves the list. Meek's
    rules run once at the end."""

    def __init__(self, score: Score):
        self.score = score

    def run(self, graph: np.ndarray, adjacency: np.ndarray, data: np.ndarray,
            c_index: np.ndarray) -> np.ndarray:
        c = len(graph) - 1
        changing = [int(v) for v in np.flatnonzero(graph[c] == 1) if np.any(graph[v] == -1)]
        result = graph.copy()
        while len(changing) > 1:
            causes = [np.union1d(np.flatnonzero(result[v, :c] == -1), np.flatnonzero(result[:c, v] == 1))
                      for v in changing]
            scores = [self.score(data[:, cause], data[:, [v]], c_index) for v, cause in zip(changing, causes)]
            best = int(np.argmin(scores))
            sink = changing.pop(best)
            result[causes[best], sink], result[sink, causes[best]] = 1, 0
        return -MeekRules.apply(-result, -adjacency)
