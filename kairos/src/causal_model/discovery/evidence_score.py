import numpy as np

from src.causal_model.discovery.node_maker import NodeMaker
from src.causal_model.nodes import Node, Scale


class EvidenceScore:
    """How believable one variable is given a set of parents: its node's evidence,
    on the scale that makes it highest. Each pair (variable, parents) is fitted once
    and remembered.

    A variable that takes a single value in the data carries no information: as
    a child every parent set scores the same (0), and as a parent it is left out.

    `intervened` gives, for each variable, True on the rows where an
    intervention set it. Its parents did not produce its value there, so those
    rows say nothing about its equation: each variable is fitted only on the
    rows where it was not set, and those rows still count for every other
    variable (the interventional score of GIES, Hauser & Bühlmann 2012).
    """

    def __init__(self, data: dict[str, np.ndarray], maker: NodeMaker,
                 intervened: dict[str, np.ndarray] | None = None):
        self.data = data
        self.maker = maker
        self.intervened = intervened if intervened is not None else {}
        self.constant = {name for name, values in data.items() if np.ptp(values) == 0}
        self.fits = 0  # how many (variable, parents) pairs were really fitted
        self._known: dict[tuple[str, frozenset[str]], tuple[float, Scale]] = {}

    def __call__(self, child: str, parents: list[str]) -> float:
        return self._best(child, parents)[0]

    def scale(self, child: str, parents: list[str]) -> Scale:
        """The scale on which the child is most believable given these parents."""
        return self._best(child, parents)[1]

    def rows(self, child: str) -> dict[str, np.ndarray]:
        """The data on the rows where `child` was not set by an intervention."""
        if child not in self.intervened or not self.intervened[child].any():
            return self.data
        kept = ~self.intervened[child]
        return {name: values[kept] for name, values in self.data.items()}

    def _best(self, child: str, parents: list[str]) -> tuple[float, Scale]:
        parents = [parent for parent in parents if parent not in self.constant]
        key = (child, frozenset(parents))
        if key not in self._known:
            self._known[key] = self._fit(child, sorted(parents))
            self.fits += 1
        return self._known[key]

    def _fit(self, child: str, parents: list[str]) -> tuple[float, Scale]:
        if child in self.constant:
            return 0.0, Scale()
        rows = self.rows(child)
        nodes: list[Node] = []
        for scale in self.maker.scales(child, rows[child]):
            node = self.maker.make(child, parents, scale)
            node.fit(rows)
            nodes.append(node)
        best = max(nodes, key=lambda node: node.evidence)
        return best.evidence, best.scale
