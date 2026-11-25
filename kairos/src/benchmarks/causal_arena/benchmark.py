import zlib
from collections.abc import Iterator
from dataclasses import dataclass

from src.benchmarks.causal_arena.data import ArenaData, ScenarioData
from src.benchmarks.causal_arena.found import FoundGraph
from src.benchmarks.causal_arena.methods import BACKGROUNDS, DiscoveryMethod
from src.benchmarks.causal_arena.scenarios import Scenario
from src.benchmarks.causal_arena.scoring import GraphScore
from src.causal_model.discovery import Arrow


@dataclass(frozen=True)
class Result:
    """One method on one file (scenario, regime, replicate) with one kind of
    background knowledge. A run that raised an error keeps the error instead of a graph."""

    scenario: Scenario
    regime: str
    replicate: int
    key: str                 # the file's name, which also seeds the random orientation
    method: str
    background: str
    truth: set[Arrow]
    found: FoundGraph | None = None
    error: str = ""

    @property
    def score(self) -> GraphScore:
        """CausalArena's protocol: undirected pairs pointed at random, one fixed seed per file."""
        return GraphScore.compare(self.graph.oriented(self.truth, "random", zlib.crc32(self.key.encode())), self.truth)

    @property
    def best(self) -> GraphScore:
        """Undirected pairs counted in their true direction."""
        return GraphScore.compare(self.graph.oriented(self.truth, "best"), self.truth)

    @property
    def worst(self) -> GraphScore:
        """Undirected pairs counted the wrong way round."""
        return GraphScore.compare(self.graph.oriented(self.truth, "worst"), self.truth)

    @property
    def graph(self) -> FoundGraph:
        if self.found is None:
            raise ValueError(f"{self.method} failed on {self.key}: {self.error}")
        return self.found


class ArenaBenchmark:
    """Every method on one file, once per kind of background knowledge it takes,
    each scored against the scenario's true graph. A method that fails is recorded
    and the others go on."""

    def __init__(self, data: ArenaData, methods: list[DiscoveryMethod]):
        self.data = data
        self.methods = methods

    def run(self, scenario: Scenario, regime: str, replicate: int) -> Iterator[Result]:
        """The results one by one, as each method finishes."""
        data = self.data.load(scenario, regime, replicate)
        for method in self.methods:
            if method.needs_change and data.change is None:
                continue
            for background in BACKGROUNDS if method.uses_background else BACKGROUNDS[:1]:
                roots = data.roots if background == "roots given" else set()
                yield self._one(data, method, background, roots)

    @staticmethod
    def _one(data: ScenarioData, method: DiscoveryMethod, background: str, roots: set[str]) -> Result:
        fixed = (data.scenario, data.regime, data.replicate, data.key, method.name, background, data.truth)
        try:
            return Result(*fixed, found=method.find(data, roots))
        except Exception as error:  # a method can fail on some data; record it and go on
            return Result(*fixed, error=f"{type(error).__name__}: {error}")
