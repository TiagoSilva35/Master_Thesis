import numpy as np

from src.causal_model.discovery.background import Background
from src.causal_model.nodes import (SCALES, BayesianFitting, BinaryNode, CategoryNode, Node, OrdinalNode,
                                    PairScreening, Scale)

LOG_ODDS_RIDGE = 1.0 / 5.0 ** 2  # a weak belief that log-odds rarely go beyond ±10 (see LaplaceNode)


class NodeMaker:
    """Makes the right node for a variable, from its kind:

        continuous    → Node (bell-shaped noise), on a plain, log or log-odds scale
        binary        → BinaryNode (logistic)
        ordinal       → OrdinalNode (ordered logit)
        nominal       → CategoryNode (softmax)

    Continuous and binary variables screen their pairs of parents for surfaces.
    The search and the model built from it use the same nodes, so they agree.
    """

    def __init__(self, background: Background, intervals: int = 8, noise_floor: float = 1e-4):
        self.background = background
        self.intervals = intervals
        self.noise_floor = noise_floor
        self.fitting = BayesianFitting()
        self.log_odds_fitting = BayesianFitting(ridge=LOG_ODDS_RIDGE)
        self.screening = PairScreening()

    def scales(self, name: str, values: np.ndarray) -> list[Scale]:
        """The scales this variable's equation may be written on."""
        if self.background.kind(name) != "continuous":
            return [SCALES[0]]
        return [scale for scale in SCALES if scale.allows(values)]

    def make(self, name: str, parents: list[str], scale: Scale | None = None) -> Node:
        kind, kinds = self.background.kind(name), self.background.kinds
        if kind == "continuous":
            return Node(name, parents, intervals=self.intervals, fitting=self.fitting, screening=self.screening,
                        noise_floor=self.noise_floor, kinds=kinds, scale=scale)
        node = {"binary": BinaryNode, "ordinal": OrdinalNode, "nominal": CategoryNode}[kind]
        return node(name, parents, intervals=self.intervals, fitting=self.log_odds_fitting,
                    screening=self.screening if kind == "binary" else None, kinds=kinds)
