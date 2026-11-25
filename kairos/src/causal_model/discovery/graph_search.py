import random

import numpy as np

from src.causal_model.discovery.background import Background
from src.causal_model.discovery.discovered_graph import DiscoveredGraph
from src.causal_model.discovery.evidence_score import EvidenceScore
from src.causal_model.discovery.graph import Arrow, Graph
from src.causal_model.discovery.node_maker import NodeMaker

Move = tuple[float, str, str, str]  # (gain, "add" / "remove" / "reverse", parent, child)


class GraphSearch:
    """Finds the parents of every variable from one table of people.

    A graph's worth is the sum of its variables' evidence, minus `price` per arrow:
    an arrow stays only if the data are more than e^price (4: about 55) times more
    probable with it.

    1. Climb: start with no arrows and keep making the single change (add, remove or
       reverse an arrow) that raises the worth most, until none does.
    2. Restart: shake the best graph (a few random arrows reversed or removed), climb
       again, and keep the result if it is worth more. Climbing alone stops at the
       first graph that no single change improves.
    3. Flag as doubtful every arrow that is worth less than `margin` more than its
       reverse.
    """

    def __init__(self, background: Background, intervals: int = 8, price: float = 4.0, margin: float = 3.0,
                 restarts: int = 20, shake: int = 6, seed: int = 0):
        self.background = background
        self.intervals = intervals
        self.price = price
        self.margin = margin
        self.restarts = restarts
        self.shake = shake
        self.seed = seed

    def run(self, data: dict[str, np.ndarray], intervened: dict[str, np.ndarray] | None = None) -> DiscoveredGraph:
        """`intervened`: for each variable, True on the rows where an intervention
        set it (see EvidenceScore); None when no row was set."""
        score = self.score(data, intervened)
        graph = self.best_graph(sorted(data), score)  # sorted, so the data's column order cannot hint at the answer
        parents = {name: list(graph.parents[name]) for name in graph.order()}
        scales = {name: score.scale(name, parents[name]) for name in parents}
        return DiscoveredGraph(parents, self.gains(graph, score), self.doubtful(graph, score), scales, self.maker())

    def score(self, data: dict[str, np.ndarray], intervened: dict[str, np.ndarray] | None = None) -> EvidenceScore:
        return EvidenceScore(data, self.maker(), intervened)

    def maker(self) -> NodeMaker:
        """Makes each variable's node from its kind (see NodeMaker)."""
        return NodeMaker(self.background, self.intervals)

    def best_graph(self, variables: list[str], score: EvidenceScore) -> Graph:
        coin = random.Random(self.seed)
        best = self.climb(Graph(variables), score)
        for _ in range(self.restarts):
            other = self.climb(self.shaken(best, coin), score)
            if self.worth(other, score) > self.worth(best, score) + 1e-6:
                best = other
        return best

    def worth(self, graph: Graph, score: EvidenceScore) -> float:
        return sum(self._value(score, child, graph.parents[child]) for child in graph.variables)

    def climb(self, graph: Graph, score: EvidenceScore) -> Graph:
        """The graph reached by always making the best single change."""
        graph = graph.copy()
        while True:
            move = max(self.moves(graph, score), default=None)
            if move is None or move[0] <= 1e-6:
                return graph
            _, kind, parent, child = move
            if kind == "add":
                graph.add(parent, child)
            else:
                graph.remove(parent, child)
                if kind == "reverse":
                    graph.add(child, parent)

    def moves(self, graph: Graph, score: EvidenceScore) -> list[Move]:
        """Every allowed single change, with how much it raises the worth."""
        moves: list[Move] = []
        for child in graph.variables:
            parents = graph.parents[child]
            current = self._value(score, child, parents)
            for other in graph.variables:
                if other in parents:
                    without = [parent for parent in parents if parent != other]
                    gain = self._value(score, child, without) - current
                    moves.append((gain, "remove", other, child))
                    if self._reversible(graph, other, child):
                        theirs = graph.parents[other]
                        gain += self._value(score, other, theirs + [child]) - self._value(score, other, theirs)
                        moves.append((gain, "reverse", other, child))
                elif self.background.allows(other, child) and not graph.creates_loop(other, child):
                    moves.append((self._value(score, child, parents + [other]) - current, "add", other, child))
        return moves

    def shaken(self, graph: Graph, coin: random.Random) -> Graph:
        """A copy with `shake` random arrows reversed or removed, half and half."""
        graph = graph.copy()
        for _ in range(min(self.shake, len(graph.arrows()))):
            parent, child = coin.choice(graph.arrows())
            reverse = coin.random() < 0.5 and self._reversible(graph, parent, child)
            graph.remove(parent, child)
            if reverse:
                graph.add(child, parent)
        return graph

    @staticmethod
    def gains(graph: Graph, score: EvidenceScore) -> dict[Arrow, float]:
        """For each arrow, how much evidence is lost when it is removed."""
        return {(parent, child): score(child, graph.parents[child])
                - score(child, [other for other in graph.parents[child] if other != parent])
                for parent, child in graph.arrows()}

    def doubtful(self, graph: Graph, score: EvidenceScore) -> list[Arrow]:
        """The arrows whose reversed graph loses less than the margin (or wins).
        Arrows the background fixes, and arrows that cannot be reversed alone
        without closing a loop, are not checked."""
        doubtful = []
        for parent, child in graph.arrows():
            if not self._reversible(graph, parent, child):
                continue
            without = [other for other in graph.parents[child] if other != parent]
            current = score(parent, graph.parents[parent]) + score(child, graph.parents[child])
            reversed_ = score(parent, graph.parents[parent] + [child]) + score(child, without)
            if current - reversed_ < self.margin:
                doubtful.append((parent, child))
        return doubtful

    def _reversible(self, graph: Graph, parent: str, child: str) -> bool:
        return self.background.allows(child, parent) and not graph.reversal_creates_loop(parent, child)

    def _value(self, score: EvidenceScore, child: str, parents: list[str]) -> float:
        return score(child, parents) - self.price * len(parents)
