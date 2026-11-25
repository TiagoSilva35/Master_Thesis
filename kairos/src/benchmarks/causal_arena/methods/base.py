from typing import Protocol

from src.benchmarks.causal_arena.data import ScenarioData
from src.benchmarks.causal_arena.found import FoundGraph

# "no background": CausalArena's protocol, every direction open.
# "roots given": the method is told which variables have no cause, taken from the
# true graph; the leaderboard's methods never get this, so it is for diagnosis only.
BACKGROUNDS = ("no background", "roots given")


class DiscoveryMethod(Protocol):
    """What the benchmark needs from a causal discovery method. To add a baseline,
    write a class with these four members and list it in run.py."""

    name: str
    uses_background: bool  # False: run once, with "no background"
    needs_change: bool     # True: run only where there are interventions (C)

    def find(self, data: ScenarioData, roots: set[str]) -> FoundGraph:
        """The graph found from the rows; no arrow may enter a variable in `roots`."""
        ...
