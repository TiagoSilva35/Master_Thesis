"""The scenarios: CausalArena's public semantic SCMs (Li, Liu, Wang and Ye, 2026).

Each scenario is a structural causal model of an operational process (a mortgage
book, a classroom, a tax office...) with 17-25 named variables. The public half has
5 scenarios in each of 10 domains, each sampled 5 times (replicates) in two regimes:
1,000 observational rows, or 800 observational and 200 interventional rows."""

import json
from dataclasses import dataclass

from src.benchmarks.causal_arena.files import ArenaFiles

MANIFEST = "scm/semantic/release_split_manifest.json"

REGIMES = {"obs_only": "observational", "obs_int": "observational + interventional"}

# The domains about decisions on people, the setting of the thesis.
PEOPLE_DOMAINS = ("education", "finance_and_credit", "government_services_and_policy_implementation",
                  "healthcare_delivery", "housing_and_real_estate")


@dataclass(frozen=True)
class Scenario:
    domain: str
    slug: str                    # e.g. mortgage_prepayment_interest_rate_shock
    directory: str               # its artifacts folder, e.g. 011_mortgage_prepayment_...
    data_files: tuple[str, ...]  # repository paths of every sampled file (both regimes, all replicates)

    def data_file(self, regime: str, replicate: int) -> str:
        ending = f"__semantic_{regime}__rep{replicate:03d}.npz"
        return next(path for path in self.data_files if path.endswith(ending))

    def replicates(self, regime: str) -> int:
        return sum(f"__semantic_{regime}__" in path for path in self.data_files)

    @property
    def graph_file(self) -> str:
        """graph.json: the variables' descriptions (measurement scale, units...) and the arrows."""
        return f"scm/semantic/artifacts/{self.domain}/{self.directory}/graph.json"


class ScenarioIndex:
    """The public scenarios, read from the release manifest."""

    @staticmethod
    def load(files: ArenaFiles) -> list[Scenario]:
        manifest = json.loads(files.path(MANIFEST).read_text())
        return [Scenario(domain, entry["data_slug"], entry["scenario_dir"],
                         tuple(f"scm/semantic/{path}" for path in entry["data_files"]))
                for domain, entries in manifest["domains"].items() for entry in entries]
