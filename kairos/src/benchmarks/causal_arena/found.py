from dataclasses import dataclass, field

import numpy as np

from src.causal_model.discovery import Arrow


@dataclass(frozen=True)
class FoundGraph:
    """What a method returned. PC and CD-NOD can leave a pair linked without a
    direction (their graphs stand for several DAGs); kairos always orients."""

    directed: set[Arrow]
    undirected: set[tuple[str, str]]  # each linked pair once, names sorted
    rows: int
    seconds: float
    doubtful: list[Arrow] = field(default_factory=list)  # kairos only

    def oriented(self, truth: set[Arrow], how: str, seed: int = 0) -> set[Arrow]:
        """Every undirected pair given a direction:
        "random": either way with equal chance, from `seed` (CausalArena's protocol:
            one fixed seed per file);
        "best": the true way when the pair is a true arrow;
        "worst": the wrong way when the pair is a true arrow.
        A pair that is no true arrow is an extra arrow whichever way it points."""
        arrows = set(self.directed)
        coin = np.random.default_rng(seed)
        for a, b in sorted(self.undirected):
            if how == "random":
                arrows.add((a, b) if coin.random() < 0.5 else (b, a))
                continue
            true_way = (a, b) if (a, b) in truth else (b, a)
            if true_way in truth:
                arrows.add(true_way if how == "best" else (true_way[1], true_way[0]))
            else:
                arrows.add((a, b))
        return arrows
