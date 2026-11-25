from itertools import combinations

import numpy as np

from src.baselines.cdnod.meek import MeekRules


class InvarianceOrientation:
    """Phase 2 of nonsta_cd_new.m, the 'generalisation of invariance'.

    CD-NOD's coding: g[i, j] = 1 with g[j, i] = 0 means i → j; -1 both ways, i — j.
      1. Every link of C (the last node) points away from it: C → x.
      2. Colliders: for x with two neighbours a and b that are not linked, and x not
         in the witness set that separated them, orient a → x ← b (C counts as a
         neighbour, which is how a changing x orients its stable neighbours).
      3. Meek's rules."""

    def run(self, skeleton: np.ndarray, witness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The oriented graph and the adjacency Meek's rules are given (the graph
        after step 1)."""
        graph = skeleton.copy()
        c = len(graph) - 1
        graph[c, graph[c] != 0] = 1
        graph[graph[:, c] != 0, c] = 0
        adjacency = graph.copy()
        for x in range(c):
            neighbours = np.flatnonzero(adjacency[:, x] != 0)
            for a, b in combinations(neighbours.tolist(), 2):
                if adjacency[a, b] != 0 or adjacency[b, a] != 0:
                    continue
                if witness[a, b, x] or witness[b, a, x]:
                    continue
                graph[a, x], graph[x, a] = 1, 0
                graph[b, x], graph[x, b] = 1, 0
        return -MeekRules.apply(-graph, -adjacency), adjacency
