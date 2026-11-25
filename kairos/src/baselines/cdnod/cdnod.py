"""CD-NOD (Huang, Zhang, Zhang, Ramsey, Sanchez-Romero, Glymour and Schölkopf, JMLR
2020), ported from the authors' MATLAB code, nonsta_cd_new.m with Type = 1
(phases 1-3: skeleton, invariance, Independent Change Principle):
https://github.com/Biwei-Huang/Causal-Discovery-from-Nonstationary-Heterogeneous-Data

Copyright (c) 2017 Biwei Huang & Kun Zhang. All rights reserved. The repository
gives no licence, so this port is for research use, with the authors credited.
Phase 4 (plots of the driving forces) is not ported: it does not change the graph."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from src.baselines.cdnod.change_score import ChangeDependence
from src.baselines.cdnod.gaussian_process import GaussianProcessWidths
from src.baselines.cdnod.independent_change import IndependentChangeOrientation
from src.baselines.cdnod.kci import IndependenceTest, KernelCITest
from src.baselines.cdnod.kci_rff import RandomFeatureCITest
from src.baselines.cdnod.minimize import ConjugateGradients
from src.baselines.cdnod.numerics import Numerics
from src.baselines.cdnod.orientation import InvarianceOrientation
from src.baselines.cdnod.settings import CDNODSettings
from src.baselines.cdnod.skeleton import SkeletonSearch, TestRecord


@dataclass(frozen=True)
class CDNODGraphs:
    """The graphs nonsta_cd_new.m returns, over the variables plus C (the last row
    and column): g_skeleton, g_inv and gns. Coding: g[i, j] = 1 with g[j, i] = 0
    means i → j; -1 both ways means i — j; 0 both ways, no link."""

    skeleton: np.ndarray
    invariance: np.ndarray
    final: np.ndarray
    tests: list[TestRecord]

    def read(self, names: list[str], graph: str = "final") -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """Arrows and undirected pairs among the variables (C left out)."""
        matrix = getattr(self, graph)
        directed, undirected = set(), set()
        for i, j in combinations(range(len(names)), 2):
            forward, backward = matrix[i, j], matrix[j, i]
            if forward == 1 and backward == 0:
                directed.add((names[i], names[j]))
            elif backward == 1 and forward == 0:
                directed.add((names[j], names[i]))
            elif forward != 0 or backward != 0:
                undirected.add((min(names[i], names[j]), max(names[i], names[j])))
        return directed, undirected

    def changing(self, names: list[str]) -> set[str]:
        """The variables with C → x: those whose mechanism changes."""
        return {names[i] for i in np.flatnonzero(self.final[-1, :-1] == 1)}


class CDNOD:

    def __init__(self, settings: CDNODSettings = CDNODSettings()):
        self.settings = settings

    def run(self, data: np.ndarray, c_index: np.ndarray) -> CDNODGraphs:
        """data: rows × variables; c_index: the time index or the domain (run) label
        of each row."""
        c_index = np.asarray(c_index, dtype=float).reshape(-1, 1)
        augmented = Numerics.standardise(np.column_stack([data, c_index]))
        random = np.random.default_rng(self.settings.seed)
        widths = GaussianProcessWidths(ConjugateGradients(length=-350))
        skeleton, witness, tests = SkeletonSearch(self._test(random, widths), self.settings.alpha,
                                                  self.settings.max_fan_in).run(augmented)
        invariance, adjacency = InvarianceOrientation().run(skeleton, witness)
        score = ChangeDependence(self.settings.width, self.settings.gp_directions, widths)
        final = IndependentChangeOrientation(score).run(invariance, adjacency, augmented, c_index)
        return CDNODGraphs(skeleton, invariance, final, tests)

    def _test(self, random: np.random.Generator, widths: GaussianProcessWidths) -> IndependenceTest:
        if self.settings.random_features:
            return RandomFeatureCITest(self.settings, random)
        return KernelCITest(self.settings, random, widths)
