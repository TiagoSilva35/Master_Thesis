from dataclasses import dataclass

from src.causal_model.discovery.graph import Arrow
from src.causal_model.discovery.node_maker import NodeMaker
from src.causal_model.kan import CausalKAN
from src.causal_model.nodes import Scale


@dataclass
class DiscoveredGraph:
    """What the search found, with the variables in causal order."""

    parents: dict[str, list[str]]  # in causal order: every variable after its parents
    gains: dict[Arrow, float]      # evidence lost when the arrow is removed (log units)
    doubtful: list[Arrow]          # arrows whose reversed direction scored about as well
    scales: dict[str, Scale]       # the scale each variable's equation is written on
    maker: NodeMaker               # makes the nodes, as the search did

    def arrows(self) -> list[Arrow]:
        return [(parent, child) for child, parents in self.parents.items() for parent in parents]

    def build(self) -> CausalKAN:
        """One node per variable with the parents and scale found, ready to fit."""
        return CausalKAN([self.maker.make(name, parents, self.scales[name])
                          for name, parents in self.parents.items()])

    def summary(self) -> str:
        """The parents of each variable, the gain of each arrow, and the flagged arrows."""
        lines = ["Gain = how much less probable the data is without the arrow (log units)."]
        for name, parents in self.parents.items():
            found = ", ".join(f"{parent} ({self.gains[(parent, name)]:.1f})" for parent in parents)
            scale = "" if self.scales[name].name == "plain" else f"  [{self.scales[name].name} scale]"
            lines.append(f"  {name} ← {found or 'no parents'}{scale}")
        doubtful = ", ".join(f"{parent} → {child}" for parent, child in self.doubtful)
        lines.append(f"Direction doubtful: {doubtful or 'none'}")
        return "\n".join(lines)
