"""Repeat drift scenarios over many seeds, to see how reliable the flags are.

drift.py runs each world once. Here every chosen world is run once per seed,
with fresh people each time, and the flags are counted:
  - a term that truly changed should be flagged in (nearly) every run;
  - a term that did not change should (nearly) never be flagged;
  - a surface should be chosen only where the world has an interaction.
The table goes to figures/worlds/repeated_summary[_<worlds>].txt. With
--figures, the first seed of each world also saves its figures in
figures/worlds/<world>/.

Run from kairos/, for example:
  PYTHONPATH=. venv/bin/python src/simulation/repeat.py                    # every world, 20 seeds
  PYTHONPATH=. venv/bin/python src/simulation/repeat.py --worlds 6 --seeds 5 --figures
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.causal_model.fitting import BayesianFitting
from src.simulation.drift import FIGURES, DriftExperiment, TermResult
from src.simulation.scenarios import SCENARIOS


def one_run(task: tuple[int, int, bool]) -> tuple[list[TermResult], dict[str, float]]:
    """One world, one seed. Top-level so worker processes can run it."""
    scenario_index, seed, save_figures = task
    scenario = SCENARIOS[scenario_index]()
    experiment = DriftExperiment(scenario, BayesianFitting(), seed=seed)
    results = experiment.run(FIGURES / scenario.world.name if save_figures else None)
    return results, experiment.chance_errors


class RepeatedDrift:
    """Runs the chosen scenarios once per seed and counts how often each term is flagged."""

    def __init__(self, scenarios: list[int], seeds: int = 20, figures: bool = False, workers: int | None = None):
        self.scenarios = scenarios  # positions in SCENARIOS
        self.seeds = seeds
        self.figures = figures
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.names = {s: SCENARIOS[s]().world.name for s in scenarios}
        # (world, term) -> (flagged, gain) for every seed where the term was in the KAN
        self.runs: dict[tuple[str, str], list[tuple[bool, float]]] = {}
        self.truly_changed: dict[tuple[str, str], bool] = {}
        self.errors: dict[str, list[dict[str, float]]] = {}  # world -> chance errors per seed

    def run(self) -> str:
        # Seeds 100, 102, ...: each run uses `seed` for the old people and `seed + 1` for the new.
        tasks = [(s, 100 + 2 * k, self.figures and k == 0) for s in self.scenarios for k in range(self.seeds)]
        with ProcessPoolExecutor(self.workers) as pool:
            for (s, _, _), (results, errors) in zip(tasks, pool.map(one_run, tasks)):
                self._add(self.names[s], results, errors)
        return self.text()

    def _add(self, world: str, results: list[TermResult], errors: dict[str, float]) -> None:
        for r in results:
            key = (world, f"{r.term} → {r.child}")
            self.truly_changed[key] = r.truly_changed
            self.runs.setdefault(key, []).append((r.flagged, r.gain))
        self.errors.setdefault(world, []).append(errors)

    def text(self) -> str:
        lines = [f"Flagged = in how many of {self.seeds} runs the term's gain was above 3 "
                 f"(its change made the data more than ≈ 20 times more probable).",
                 "A surface counts as not flagged in runs where it was not chosen.", "",
                 f"{'world':30} {'term':24} {'truly changed':>14} {'flagged':>9} {'median gain':>12}"]
        for (world, term), runs in self.runs.items():
            truly = "yes" if self.truly_changed[(world, term)] else "no"
            flagged = sum(f for f, _ in runs)
            lines.append(f"{world:30} {term:24} {truly:>14} {f'{flagged}/{self.seeds}':>9} "
                         f"{np.median([g for _, g in runs]):12.1f}")
        lines += ["", "How often each surface was chosen:"]
        lines += [f"  {world} {term}: {len(runs)}/{self.seeds}" for (world, term), runs in self.runs.items() if "×" in term]
        lines += ["", "Error in P(Y=1) on people from after the drift (median over seeds):"]
        for world, errors in self.errors.items():
            old = np.median([e["old model, people from after the drift"] for e in errors])
            new = np.median([e["updated model, people from after the drift"] for e in errors])
            lines.append(f"  {world}: old model {old:.3f}, updated model {new:.3f}")
        lines += ["", f"Totals: {self._totals()}"]
        return "\n".join(lines) + "\n"

    def _totals(self) -> str:
        caught = missed = alarms = quiet = 0
        for key, runs in self.runs.items():
            flagged = sum(f for f, _ in runs)
            if self.truly_changed[key]:
                caught, missed = caught + flagged, missed + self.seeds - flagged
            else:
                alarms, quiet = alarms + flagged, quiet + self.seeds - flagged
        return f"real changes caught {caught}/{caught + missed}; false alarms {alarms}/{alarms + quiet} term-runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worlds", nargs="*", default=None, help="world name prefixes, e.g. 4 6 (default: every world)")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--figures", action="store_true", help="save figures for the first seed")
    args = parser.parse_args()
    chosen = [s for s, make in enumerate(SCENARIOS)
              if args.worlds is None or any(make().world.name.startswith(p) for p in args.worlds)]
    text = RepeatedDrift(chosen, seeds=args.seeds, figures=args.figures).run()
    suffix = "" if args.worlds is None else "_" + "_".join(args.worlds)
    (FIGURES / f"repeated_summary{suffix}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
