"""PC and CD-NOD from causal-learn (CMU / PyWhy) as baselines.

Independence tests: "fisherz" assumes straight-line relations with bell-shaped
noise and is fast; "rcit" (randomized conditional independence test) handles
curved relations and scales to many rows, unlike the kernel test "kci", whose
cost grows with the cube of the number of rows.
"""

import time
from itertools import combinations

import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge

from src.benchmarks.causal_arena.data import ScenarioData
from src.benchmarks.causal_arena.found import FoundGraph
from src.causal_model.discovery import Arrow

TEST_NAMES = {"fisherz": "Fisher z", "rcit": "RCIT"}


class CausalLearnGraph:
    """Translation between our variable names and causal-learn's graphs, whose
    nodes are called X1, X2, ... in column order."""

    @staticmethod
    def read(matrix: np.ndarray, names: list[str]) -> tuple[set[Arrow], set[tuple[str, str]]]:
        """graph[j, i] = 1 and graph[i, j] = -1 means i → j; both -1 (undirected)
        or both 1 (both ways) mean linked without a direction. Columns beyond
        `names` (CD-NOD's changing factor, the last node) are left out."""
        directed, undirected = set(), set()
        for i, j in combinations(range(len(names)), 2):
            forward, backward = matrix[i, j], matrix[j, i]
            if forward == -1 and backward == 1:
                directed.add((names[i], names[j]))
            elif forward == 1 and backward == -1:
                directed.add((names[j], names[i]))
            elif forward != 0 and backward != 0:
                undirected.add((min(names[i], names[j]), max(names[i], names[j])))
        return directed, undirected

    @staticmethod
    def no_arrow_into(protected: set[str], names: list[str]) -> BackgroundKnowledge | None:
        """Background knowledge forbidding every arrow into the protected variables."""
        if not protected:
            return None
        nodes = {name: GraphNode(f"X{index + 1}") for index, name in enumerate(names)}
        knowledge = BackgroundKnowledge()
        for root in protected:
            for other in names:
                if other != root:
                    knowledge.add_forbidden_by_node(nodes[other], nodes[root])
        return knowledge


class PCSearch:
    """The PC algorithm: cut a link when a test finds the two variables independent
    given others, then orient colliders and Meek's rules. Interventional rows, when
    present, are pooled with the rest."""

    uses_background = True
    needs_change = False

    def __init__(self, test: str, alpha: float = 0.05):
        self.test = test
        self.alpha = alpha
        self.name = f"PC ({TEST_NAMES[test]})"

    def find(self, data: ScenarioData, roots: set[str]) -> FoundGraph:
        knowledge = CausalLearnGraph.no_arrow_into(roots, data.variables)
        np.random.seed(0)  # RCIT draws random features
        start = time.perf_counter()
        graph = pc(data.values, self.alpha, self.test, background_knowledge=knowledge, show_progress=False)
        seconds = time.perf_counter() - start
        directed, undirected = CausalLearnGraph.read(graph.G.graph, data.variables)
        return FoundGraph(directed, undirected, len(data.values), seconds)


class CDNODSearch:
    """CD-NOD (Huang et al., 2020) as causal-learn implements it: PC on the
    variables plus C (here: which intervention produced the row, 0 for none), every
    link of C pointing away from it (steps 1-2 of the paper; causal-learn has no
    Independent Change Principle)."""

    uses_background = True
    needs_change = True

    def __init__(self, test: str, alpha: float = 0.05):
        self.test = test
        self.alpha = alpha
        self.name = f"CD-NOD ({TEST_NAMES[test]})"

    def find(self, data: ScenarioData, roots: set[str]) -> FoundGraph:
        assert data.change is not None, "CD-NOD needs interventions"
        # RCIT fails (its eigenvalues do not converge) on a variable with only a few
        # distinct values, so each label gets a small jitter, drawn independently of
        # everything else; the labels stay apart (±0.25).
        factor = data.change + np.random.default_rng(0).uniform(-0.25, 0.25, len(data.change))
        knowledge = CausalLearnGraph.no_arrow_into(roots, data.variables)
        np.random.seed(0)
        start = time.perf_counter()
        graph = self.search(data.values, factor, self.alpha, self.test, knowledge)
        seconds = time.perf_counter() - start
        directed, undirected = CausalLearnGraph.read(graph.G.graph, data.variables)
        return FoundGraph(directed, undirected, len(data.values), seconds)

    @staticmethod
    def search(matrix: np.ndarray, factor: np.ndarray, alpha: float, test: str,
               knowledge: BackgroundKnowledge | None) -> CausalGraph:
        """causal-learn's cdnod (version 0.1.4.8, default rules) with one change:
        the skeleton search also gets the background knowledge, as it does in
        causal-learn's pc. cdnod leaves it out there, so two roots that are
        correlated stay linked and the link is then pointed into one of them."""
        augmented = np.column_stack([matrix, factor])  # the changing factor is the last node
        graph = SkeletonDiscovery.skeleton_discovery(augmented, alpha, CIT(augmented, test),
                                                     background_knowledge=knowledge, show_progress=False)
        factor_node = graph.G.nodes[-1]
        for node in graph.G.get_adjacent_nodes(factor_node):  # the factor can only be a cause
            graph.G.add_directed_edge(factor_node, node)
        if knowledge is not None:
            orient_by_background_knowledge(graph, knowledge)
        graph = UCSepset.uc_sepset(graph, 2, background_knowledge=knowledge)  # colliders
        return Meek.meek(graph, background_knowledge=knowledge)  # arrows that follow from them
