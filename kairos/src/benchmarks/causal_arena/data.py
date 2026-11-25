import json
from dataclasses import dataclass

import numpy as np

from src.benchmarks.causal_arena.files import ArenaFiles
from src.benchmarks.causal_arena.scenarios import Scenario
from src.causal_model.discovery import Arrow


@dataclass(frozen=True)
class ScenarioData:
    """One replicate of one scenario in one regime: the rows every method gets,
    and the true graph they are scored against."""

    scenario: Scenario
    regime: str
    replicate: int
    variables: list[str]
    values: np.ndarray          # rows × variables, standardised as released (mean 0, spread 1)
    raw: np.ndarray             # the same in original units: category codes, counts, dollars...
    kinds: dict[str, str]       # each variable's measurement scale: continuous, binary, ordinal_categorical, ...
    intervened: np.ndarray      # rows × variables, True where an intervention set the value
    truth: set[Arrow]
    change: np.ndarray | None   # C: 0 for observational rows, k for rows of the k-th intervention; None if none
    change_kind: str            # "intervention" or "none"

    @property
    def key(self) -> str:
        """Names the file; also seeds the random orientation of undirected pairs."""
        return f"{self.scenario.domain}/{self.scenario.slug}/{self.regime}/rep{self.replicate}"

    @property
    def roots(self) -> set[str]:
        """The variables with no cause in the true graph."""
        return set(self.variables) - {effect for _, effect in self.truth}

    def table(self) -> dict[str, np.ndarray]:
        """The standardised values as one array per variable, the form kairos takes."""
        return {name: self.values[:, column] for column, name in enumerate(self.variables)}


class ArenaData:
    """Reads a sampled file. Its arrays: x (rows × variables × 2: the standardised
    value, and 1 where an intervention set it), g (the true graph, row = cause),
    node_ids, and metadata_json (with the mean and spread used to standardise)."""

    def __init__(self, files: ArenaFiles):
        self.files = files

    def load(self, scenario: Scenario, regime: str, replicate: int) -> ScenarioData:
        with np.load(self.files.path(scenario.data_file(regime, replicate)), allow_pickle=False) as archive:
            x, graph = archive["x"].astype(float), archive["g"]
            variables = [str(name) for name in archive["node_ids"]]
            metadata = json.loads(str(archive["metadata_json"]))
        values, intervened = x[..., 0], x[..., 1] > 0
        scale = metadata["standardization"]
        raw = values * np.array([scale[name]["std"] for name in variables]) + np.array(
            [scale[name]["mean"] for name in variables])
        truth = {(variables[cause], variables[effect]) for cause, effect in zip(*np.nonzero(graph))}
        change = self._change(intervened)
        return ScenarioData(scenario, regime, replicate, variables, values, raw, self._kinds(scenario),
                            intervened, truth, change, "none" if change is None else "intervention")

    def _kinds(self, scenario: Scenario) -> dict[str, str]:
        nodes = json.loads(self.files.path(scenario.graph_file).read_text())["nodes"]
        return {node["id"]: node["measurement_scale"] for node in nodes}

    @staticmethod
    def _change(intervened: np.ndarray) -> np.ndarray | None:
        """0 for observational rows; each distinct set of intervened variables
        gets its own number, 1, 2, ..., in a fixed order."""
        if not intervened.any():
            return None
        codes = intervened.astype(np.int64) @ (2 ** np.arange(intervened.shape[1], dtype=np.int64))
        targets = np.unique(codes[codes > 0])
        return np.searchsorted(targets, codes).astype(float) + (codes > 0)
