import time

from src.baselines.cdnod import CDNOD, CDNODSettings
from src.benchmarks.causal_arena.data import ScenarioData
from src.benchmarks.causal_arena.found import FoundGraph


class AuthorsCDNODSearch:
    """CD-NOD as the authors' MATLAB code runs it (src/baselines/cdnod: steps 1-3
    with the Independent Change Principle), C = which intervention produced the row
    (1 for none, 2, 3, ... as their domain labels). Their code takes no background
    knowledge, and their rule for 1,000 rows learns kernel widths with a Gaussian
    process in every test: several hours per file, so it is not in the default run."""

    name = "CD-NOD (authors')"
    uses_background = False
    needs_change = True

    def __init__(self, seed: int = 0):
        self.seed = seed

    def find(self, data: ScenarioData, roots: set[str]) -> FoundGraph:
        assert data.change is not None, "CD-NOD needs interventions"
        settings = CDNODSettings.for_rows(len(data.values), seed=self.seed)
        start = time.perf_counter()
        graphs = CDNOD(settings).run(data.values, data.change + 1)
        seconds = time.perf_counter() - start
        directed, undirected = graphs.read(data.variables)
        return FoundGraph(directed, undirected, len(data.values), seconds)
