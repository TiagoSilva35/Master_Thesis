from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.baselines.cdnod.kci import IndependenceTest


@dataclass(frozen=True)
class TestRecord:
    """One independence test (nonsta_cd_new.m keeps these as SP)."""

    i: int
    j: int
    given: tuple[int, ...]
    p_value: float


class SkeletonSearch:
    """Phase 1 of nonsta_cd_new.m: start with every pair of columns linked (C, the
    changing factor, is the last column) and cut a link when the test finds the two
    independent, given no variable, then one, then two (up to max_fan_in) of the
    first one's current neighbours.

    As in MATLAB: links are cut as soon as a test says so (not the order-free
    'stable' PC), each pair is tested from both ends, and the testing of a pair goes
    on after its link is cut, so the 'witness' set that separates two variables is
    the union of every set that did."""

    def __init__(self, test: IndependenceTest, alpha: float, max_fan_in: int):
        self.test = test
        self.alpha = alpha
        self.max_fan_in = max_fan_in

    def run(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[TestRecord]]:
        """The skeleton (-1 where linked), the witness sets (witness[i, j, k]: k helped
        separate i and j) and every test run."""
        n = data.shape[1]
        graph = np.eye(n) - np.ones((n, n))
        witness = np.zeros((n, n, n), dtype=bool)
        records: list[TestRecord] = []
        for size in range(self.max_fan_in + 1):
            for i in range(n):
                neighbours = np.flatnonzero(graph[i] != 0)
                if len(neighbours) <= size:
                    continue
                for j in neighbours:
                    others = neighbours[neighbours != j]
                    for given in combinations(others.tolist(), size):
                        p_value = self.test.p_value(data[:, [i]], data[:, [j]], data[:, list(given)])
                        records.append(TestRecord(i, int(j), given, p_value))
                        if p_value > self.alpha:
                            witness[i, j, list(given)] = True
                            witness[j, i, list(given)] = True
                            graph[i, j] = graph[j, i] = 0
        return graph, witness, records
