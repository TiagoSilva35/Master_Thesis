"""Simulate a drift in several worlds and see which terms the causal KAN says changed.

For every scenario in scenarios.py: draw people from before and after the drift,
fit every variable once on both windows with ChangeFit (shared parts plus change
parts, see change.py), and check the terms it says changed against what truly
changed. Saves, in figures/worlds/<world name>/:
  before.png      the old model, fitted on people from before the drift only
  after.png       the updated model (shared plus change parts of the joint fit)
  comparison.png  each curve before and after the drift, from the joint fit, with
                  95% bands and the true curves; for each surface, how it changed
  summary.txt     each term's gain and whether it counts as changed, and how
                  wrong the old model's chances become

Run from kairos/:  PYTHONPATH=. venv/bin/python src/simulation/drift.py
For many seeds at once, see repeat.py.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.causal_model.change import ChangeFit
from src.causal_model.fitting import BayesianFitting
from src.causal_model.kan import CausalKAN
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.causal_model.plot import KANPlot
from src.causal_model.screening import PairScreening
from src.simulation.scenarios import SCENARIOS, DriftScenario
from src.simulation.world import World

FIGURES = Path(__file__).resolve().parents[2] / "figures" / "worlds"


@dataclass
class TermResult:
    term: str             # a parent, or a pair like "income×savings"
    child: str
    truly_changed: bool
    gain: float           # how much more probable the data is with this term's change on (log units)
    flagged: bool         # gain above the margin: the KAN says this term changed


class DriftPlot:
    """One panel per curve: the joint fit's curve before (blue) and after
    (orange) the drift, each with its 95% band, and the true curves dashed when
    the world has them. One heat map per surface: how much it changed. Each
    title gives the term's gain."""

    def __init__(self, scenario: DriftScenario, before: CausalKAN, after: CausalKAN,
                 gains: dict[tuple[str, str], float], pooled: dict[str, np.ndarray],
                 points: int = 200, columns: int = 3):
        self.scenario = scenario
        self.before = before
        self.after = after
        self.gains = gains    # (term, child) -> gain
        self.pooled = pooled  # everyone's values: the joint fit centred its curves on them
        self.points = points
        self.columns = columns

    def figure(self):
        panels = list(self.scenario.world.edges())
        panels += [(pair, node.name) for node in self.after.nodes for pair in node.pairs]
        columns = min(self.columns, len(panels))
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows), squeeze=False)
        for ax, (term, child) in zip(axes.flat, panels):
            if isinstance(term, tuple):
                self._draw_surface(fig, ax, term, child)
            else:
                self._draw_edge(ax, term, child)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(self.scenario.description, fontsize=10)
        fig.tight_layout()
        return fig

    def _draw_edge(self, ax, parent: str, child: str) -> None:
        for model, world, colour, label in ((self.before, self.scenario.world, "C0", "before"),
                                            (self.after, self.scenario.drifted, "C1", "after")):
            curve = model.node(child).curves[parent]
            x = np.linspace(curve.knots[0], curve.knots[-1], self.points)
            low, high = curve.band(x)
            ax.fill_between(x, low, high, color=colour, alpha=0.2, linewidth=0)
            ax.plot(x, curve(x), color=colour, label=label)
            truth = world.true_curve(child, parent)
            if truth is not None:
                ax.plot(x, truth(x) - truth(self.pooled[parent]).mean(), color=colour, linestyle="--",
                        linewidth=1, label=f"truth {label}")
        ax.set_title(self._title(f"{parent} → {child}", parent, child), fontsize=9)
        ax.set_xlabel(parent)
        ax.axhline(0.0, color="grey", linewidth=0.5)

    def _draw_surface(self, fig, ax, pair: tuple[str, str], child: str) -> None:
        old, new = self.before.node(child).surfaces[pair], self.after.node(child).surfaces[pair]
        grid1, grid2 = np.meshgrid(np.linspace(old.first.knots[0], old.first.knots[-1], 60),
                                   np.linspace(old.second.knots[0], old.second.knots[-1], 60))
        change = (new(grid1.ravel(), grid2.ravel()) - old(grid1.ravel(), grid2.ravel())).reshape(grid1.shape)
        limit = max(float(np.max(np.abs(change))), 1e-3)
        mesh = ax.pcolormesh(grid1, grid2, change, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        fig.colorbar(mesh, ax=ax, label="after − before (push)")
        label = Node.surface_label(pair)
        ax.set_title(self._title(f"{pair[0]} × {pair[1]} → {child}", label, child), fontsize=9)
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])

    def _title(self, name: str, term: str, child: str) -> str:
        truly = "truly changed" if (term, child) in self.scenario.changed_edges else "not changed"
        return f"{name}  ({truly}, gain {self.gains[(term, child)]:.1f})"


class DriftExperiment:
    """Draws people from before and after a drift, fits the old model and the
    joint fit, and reports which terms changed.

    Every node with at least two parents chooses its surfaces with a
    PairScreening. A term counts as changed when its change part makes the data
    more than e^margin times more probable (margin 3 ≈ 20 times).
    """

    def __init__(self, scenario: DriftScenario, fitting: BayesianFitting, intervals: int = 16,
                 size: int = 10_000, margin: float = 3.0, seed: int = 0):
        self.scenario = scenario
        self.fitting = fitting
        self.intervals = intervals
        self.size = size
        self.margin = margin
        self.seed = seed
        self.screening = PairScreening()
        self.levels: dict[str, float] = {}         # each node's gain for a change in level
        self.chance_errors: dict[str, float] = {}  # error in P(Y=1) against the true chance

    def run(self, directory: Path | None = None) -> list[TermResult]:
        """Fit and compare; save figures and summary when `directory` is given."""
        old_people = self.scenario.world.sample(self.size, self.seed)
        new_people = self.scenario.drifted.sample(self.size, self.seed + 1)
        old_model = self._fitted_model(self.scenario.world, old_people)
        before, after, results = self._joint(old_people, new_people)
        self.chance_errors = {
            "old model, people from before the drift": self._chance_error(old_model, self.scenario.world, old_people),
            "old model, people from after the drift": self._chance_error(old_model, self.scenario.drifted, new_people),
            "updated model, people from after the drift": self._chance_error(after, self.scenario.drifted, new_people),
        }
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            pooled = {name: np.concatenate([old_people[name], new_people[name]]) for name in old_people}
            gains = {(r.term, r.child): r.gain for r in results}
            self._save(KANPlot(old_model).figure(), directory / "before.png")
            self._save(KANPlot(after).figure(), directory / "after.png")
            self._save(DriftPlot(self.scenario, before, after, gains, pooled).figure(), directory / "comparison.png")
            (directory / "summary.txt").write_text(self._summary(results, after))
        return results

    def _node(self, world: World, name: str, parents: list[str]) -> Node:
        kind = OutcomeNode if name == world.outcome else Node
        screening = self.screening if len(parents) >= 2 else None
        return kind(name, parents, pairs=[], intervals=self.intervals, fitting=self.fitting, screening=screening)

    def _fitted_model(self, world: World, people: dict) -> CausalKAN:
        model = CausalKAN([self._node(world, name, parents) for name, parents in world.parents().items()])
        model.fit(people)
        return model

    def _joint(self, old_people: dict, new_people: dict) -> tuple[CausalKAN, CausalKAN, list[TermResult]]:
        """One joint fit per node with parents; a root is fitted on each window alone."""
        change = ChangeFit(margin=self.margin)
        before_nodes, after_nodes, results = [], [], []
        for name, parents in self.scenario.world.parents().items():
            if not parents:
                for nodes, world, people in ((before_nodes, self.scenario.world, old_people),
                                             (after_nodes, self.scenario.drifted, new_people)):
                    node = self._node(world, name, parents)
                    node.fit(people)
                    nodes.append(node)
                continue
            outcome = change.fit(self._node(self.scenario.world, name, parents), old_people, new_people)
            before_nodes.append(outcome.before)
            after_nodes.append(outcome.after)
            self.levels[name] = outcome.gains["level"]
            for term, gain in outcome.gains.items():
                if term != "level":
                    truly = (term, name) in self.scenario.changed_edges
                    results.append(TermResult(term, name, truly, gain, gain > self.margin))
        return CausalKAN(before_nodes), CausalKAN(after_nodes), results

    def _summary(self, results: list[TermResult], after: CausalKAN) -> str:
        surfaces = {node.name: node.pairs for node in after.nodes if node.pairs}
        lines = [f"World: {self.scenario.world.name}",
                 f"Drift: {self.scenario.description}",
                 f"Surfaces chosen: {surfaces or 'none'}",
                 "Gain = how much more probable the data is with that term's change on (log units).",
                 f"Change in level, gain per node: { {k: round(v, 1) for k, v in self.levels.items()} }",
                 "",
                 f"{'term':26} {'truly changed':>14} {'gain':>10} {'flagged (gain > ' + str(self.margin) + ')':>22}"]
        for r in results:
            lines.append(f"{r.term + ' → ' + r.child:26} {'yes' if r.truly_changed else 'no':>14} "
                         f"{r.gain:10.1f} {'yes' if r.flagged else 'no':>22}")
        lines += ["", "Error in P(Y=1) against the true chance, averaged over people:"]
        lines += [f"  {who:44} {error:.3f}" for who, error in self.chance_errors.items()]
        return "\n".join(lines + [""])

    @staticmethod
    def _chance_error(model: CausalKAN, world: World, people: dict) -> float:
        predicted = model.node(world.outcome).predict(people)
        return float(np.mean(np.abs(predicted - world.chance(people))))

    @staticmethod
    def _save(fig, path: Path) -> None:
        fig.savefig(path, dpi=110)
        plt.close(fig)


def main() -> None:
    for make_scenario in SCENARIOS:
        scenario = make_scenario()
        experiment = DriftExperiment(scenario, BayesianFitting())
        experiment.run(FIGURES / scenario.world.name)
        print((FIGURES / scenario.world.name / "summary.txt").read_text())
    print(f"figures saved in {FIGURES}")


if __name__ == "__main__":
    main()
