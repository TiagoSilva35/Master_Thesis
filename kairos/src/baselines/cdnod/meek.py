"""meeks.m (adapted from Kevin Murphy's Bayes Net Toolbox; Copyright (C) 1997-2002
Kevin Murphy, 2010-2011 Robert Tillman; GNU General Public License v3 or later).
This port is a derivative of that file and carries the same licence."""

import numpy as np


class MeekRules:
    """Meek's four rules, applied until nothing changes.

    Coding of `pdag` (the opposite sign of CD-NOD's): -1 at (a, b) with 0 at (b, a)
    means a → b; 1 at both means a — b. `adjacency` is the graph before any
    collider was oriented, coded the same way.

    As in MATLAB, each rule walks the edges it found at its start in MATLAB's
    column-by-column order and checks each against the graph as it is being
    changed, so the order of the edges can matter. Rule 4 keeps the original's
    simplified condition (it does not check that d and b are not linked)."""

    @staticmethod
    def apply(pdag: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        pdag = pdag.copy()
        previous = np.zeros_like(pdag)
        while not np.array_equal(pdag, previous):
            previous = pdag.copy()
            for a, b in MeekRules._find(pdag == -1):  # rule 1: a → b — c, a and c apart: b → c
                c = np.flatnonzero((pdag[b] == 1) & (adjacency[a] == 0))
                if len(c):
                    pdag[b, c], pdag[c, b] = -1, 0
            for a, b in MeekRules._find(pdag == 1):  # rule 2: a → c → b and a — b: a → b
                if np.any((pdag[a] == -1) & (pdag[:, b] == -1)):
                    pdag[a, b], pdag[b, a] = -1, 0
            for a, b in MeekRules._find(pdag == 1):  # rule 3: two unlinked c with a — c → b
                c = np.flatnonzero((adjacency[a] == 1) & (pdag[:, b] == -1))
                among = adjacency[np.ix_(c, c)].copy()
                np.fill_diagonal(among, 1)
                if np.any(among == 0):
                    pdag[a, b], pdag[b, a] = -1, 0
            for a, b in MeekRules._find(pdag == 1):  # rule 4: d → c → b, c — a, d — a
                for c in np.flatnonzero((pdag[:, b] == -1) & (adjacency[:, a] == 1)):
                    if np.any((pdag[:, c] == -1) & (pdag[:, a] == 1)):
                        pdag[a, b], pdag[b, a] = -1, 0
        return pdag

    @staticmethod
    def _find(mask: np.ndarray) -> list[tuple[int, int]]:
        """MATLAB's [A, B] = find(mask): (row, column) pairs, column by column."""
        columns, rows = np.nonzero(mask.T)
        return [(int(row), int(column)) for row, column in zip(rows, columns)]
