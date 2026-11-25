import time

from src.benchmarks.causal_arena.data import ScenarioData
from src.benchmarks.causal_arena.found import FoundGraph
from src.benchmarks.causal_arena.methods.kairos_inputs import KairosInputs
from src.causal_model.discovery import Background, GraphSearch


class KairosSearch:
    """Kairos' GraphSearch (src/causal_model/discovery) on one file. Each variable's
    kind (continuous, binary, ordinal or nominal) is read from its measurement
    scale by KairosInputs and handed to the search in the background. In obs_int
    files each variable is fitted only on the rows where no intervention set it."""

    name = "kairos"
    uses_background = True
    needs_change = False

    def find(self, data: ScenarioData, roots: set[str]) -> FoundGraph:
        inputs = KairosInputs(data)
        search = GraphSearch(Background(roots=roots, kinds=inputs.kinds))
        start = time.perf_counter()
        found = search.run(inputs.table(), inputs.intervened())
        seconds = time.perf_counter() - start
        return FoundGraph(set(found.arrows()), set(), len(data.values), seconds, list(found.doubtful))
