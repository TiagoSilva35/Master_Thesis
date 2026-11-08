"""Step 3: the whole graph."""

import numpy as np

from src.causal_model.node import Node


class CausalKAN:
    def __init__(self, nodes: list[Node]):
        self._check_order(nodes)
        self.nodes = nodes
        self.finder = {node.name: node for node in nodes}

    def fit(self, data: dict[str, np.ndarray]) -> None:
        for node in self.nodes:
            node.fit(data)

    def node(self, name: str) -> Node:
        """Get the node with this name."""
        found = self.finder.get(name)
        if found is None:
            raise ValueError(f"no node named {name}")
        return found

    @staticmethod
    def _check_order(nodes: list[Node]) -> None:
        seen = set()
        for node in nodes:
            missing = [parent for parent in node.parents if parent not in seen]
            if missing:
                raise ValueError(f"{node.name} comes before its parents {missing}")
            seen.add(node.name)
