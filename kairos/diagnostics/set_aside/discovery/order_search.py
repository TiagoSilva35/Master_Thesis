import random

from src.causal_model.discovery.background import Background
from src.causal_model.discovery.evidence_score import EvidenceScore
from src.causal_model.discovery.graph import Graph
from src.causal_model.discovery.parent_sets import ParentSets


class OrderSearch:
    """A search over orders of the variables instead of over arrows (Teyssier &
    Koller 2005). An order says who may be a parent of whom: each variable takes
    its best parent set among the variables before it (ParentSets), so every
    order gives a graph without loops, and the score of an order is the sum of
    those best evidences.

    The search climbs by swapping two neighbours in the order. A swap changes the
    options of those two variables only, so it costs two look-ups. Each step takes
    the best swap that is not tabu (one of the last `tabu` swaps undone), even when
    it lowers the score, so the search can walk out of a local top; it stops after
    `patience` steps without beating the best order so far. Then it restarts
    `restarts` times from the best order shaken by `shake` random swaps.

    It starts from the greedy search's graph: its order, and its parent sets added
    as options, so the result never scores below greedy's.

    `candidates` and `most` shape the parent sets (see ParentSets): parents come
    from each variable's `candidates` strongest single arrows; every set of up to
    `most` of them is scored, and larger sets are grown one parent at a time.
    """

    def __init__(self, candidates: int = 10, most: int = 2, tabu: int = 10, patience: int = 20,
                 restarts: int = 5, shake: int = 10, seed: int = 0):
        self.candidates = candidates
        self.most = most
        self.tabu = tabu
        self.patience = patience
        self.restarts = restarts
        self.shake = shake
        self.seed = seed

    def improve(self, start: Graph, score: EvidenceScore, background: Background) -> Graph:
        """A graph at least as good as `start` under `score`."""
        sets = ParentSets(score, background, self.candidates, self.most)
        sets.prepare(start.variables, {name: list(start.parents[name]) for name in start.variables})
        order = self.search(start.order(), sets, background)
        return self.graph(order, sets)

    def search(self, start: list[str], sets: ParentSets, background: Background) -> list[str]:
        coin = random.Random(self.seed)
        movable = [name for name in start if name != background.outcome]
        last = [background.outcome] if background.outcome in start else []
        best = movable
        best_total = self.total(best + last, sets)
        for restart in range(self.restarts + 1):
            order = list(best)
            if restart > 0:
                for _ in range(self.shake):
                    i = coin.randrange(len(order) - 1)
                    order[i], order[i + 1] = order[i + 1], order[i]
            order, total = self._climb(order, last, sets)
            if total > best_total:
                best, best_total = order, total
        return best + last

    def total(self, order: list[str], sets: ParentSets) -> float:
        return sum(sets.best(name, set(order[:i]))[0] for i, name in enumerate(order))

    def graph(self, order: list[str], sets: ParentSets) -> Graph:
        graph = Graph(sorted(order))
        for i, name in enumerate(order):
            for parent in sets.best(name, set(order[:i]))[1]:
                graph.add(parent, name)
        return graph

    def _climb(self, order: list[str], last: list[str], sets: ParentSets) -> tuple[list[str], float]:
        """Best-swap steps with a tabu list; returns the best order met and its score."""
        current = self.total(order + last, sets)
        best, best_total = list(order), current
        tabu: list[frozenset[str]] = []
        waited = 0
        while waited < self.patience and len(order) > 1:
            move, change = None, float("-inf")
            for i in range(len(order) - 1):
                pair = frozenset(order[i:i + 2])
                if pair in tabu:
                    continue
                delta = self._swap_change(order, i, sets)
                if delta > change:
                    move, change = i, delta
            if move is None:
                break
            order[move], order[move + 1] = order[move + 1], order[move]
            current += change
            tabu = (tabu + [frozenset(order[move:move + 2])])[-self.tabu:]
            if current > best_total + 1e-9:
                best, best_total, waited = list(order), current, 0
            else:
                waited += 1
        return best, best_total

    @staticmethod
    def _swap_change(order: list[str], i: int, sets: ParentSets) -> float:
        """How the score changes when order[i] and order[i+1] swap: the second
        loses the first as a possible parent, the first gains the second."""
        first, second = order[i], order[i + 1]
        before = set(order[:i])
        old = sets.best(first, before)[0] + sets.best(second, before | {first})[0]
        new = sets.best(second, before)[0] + sets.best(first, before | {second})[0]
        return new - old
