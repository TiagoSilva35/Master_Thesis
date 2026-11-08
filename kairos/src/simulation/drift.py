"""Fit a causal KAN before and after a drift, in several worlds, and see how it reacts.

For every scenario in scenarios.py this saves, in figures/worlds/<world name>/:
  before.png      the KAN fitted on people from before the drift
  after.png       the KAN fitted on people from after the drift
  comparison.png  each edge's curve before and after, with bands and the true curves
  summary.txt     which edges look changed, and how wrong the old model's chances become

The KAN is refitted from scratch after the drift (no online updates yet).

Run from kairos/:  PYTHONPATH=. venv/bin/python src/simulation/drift.py
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.causal_model.fitting import BayesianFitting, Fitting
from src.causal_model.kan import CausalKAN
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.causal_model.plot import KANPlot
from src.causal_model.spline import Spline
from src.simulation.scenarios import SCENARIOS, DriftScenario
from src.simulation.world import World

FIGURES = Path(__file__).resolve().parents[2] / "figures" / "worlds"


class EdgeComparison:
    """Compares two fits of the same edge on the range both have data for.

    A curve's level is not pinned down (the node's constant takes the rest), so
    both curves are shifted to average zero over the shared range first. The
    gap is the largest distance between them, counted in combined spreads.
    """

    def __init__(self, points: int = 200):
        self.points = points

    def shared_range(self, before: Spline, after: Spline) -> np.ndarray | None:
        low = max(before.knots[0], after.knots[0])
        high = min(before.knots[-1], after.knots[-1])
        return np.linspace(low, high, self.points) if low < high else None

    @staticmethod
    def levelled(curve: Spline, x: np.ndarray, reference: np.ndarray):
        """The curve and its spread at x, shifted to average zero over `reference`."""
        rows = curve.weights(x) - curve.weights(reference).mean(axis=0)
        values = rows @ curve.heights
        spreads = np.sqrt(np.maximum(np.sum((rows @ curve.covariance) * rows, axis=1), 0.0))
        return values, spreads

    def gap(self, before: Spline, after: Spline) -> float:
        x = self.shared_range(before, after)
        if x is None:
            return float("nan")
        old, old_spread = self.levelled(before, x, x)
        new, new_spread = self.levelled(after, x, x)
        return float(np.max(np.abs(new - old) / np.sqrt(old_spread**2 + new_spread**2 + 1e-12)))


class DriftPlot:
    """One panel per edge: the curve before (blue) and after (orange) the drift,
    each with its 95% band, and the true curves dashed when the world has them."""

    def __init__(self, scenario: DriftScenario, before: CausalKAN, after: CausalKAN,
                 comparison: EdgeComparison, columns: int = 3):
        self.scenario = scenario
        self.before = before
        self.after = after
        self.comparison = comparison
        self.columns = columns

    def figure(self):
        edges = self.scenario.world.edges()
        columns = min(self.columns, len(edges))
        rows = math.ceil(len(edges) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows), squeeze=False)
        for ax, (parent, child) in zip(axes.flat, edges):
            self._draw_edge(ax, parent, child)
        for ax in axes.flat[len(edges):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(self.scenario.description, fontsize=10)
        fig.tight_layout()
        return fig

    def _draw_edge(self, ax, parent: str, child: str) -> None:
        old = self.before.node(child).curves[parent]
        new = self.after.node(child).curves[parent]
        reference = self.comparison.shared_range(old, new)
        if reference is None:
            reference = np.linspace(old.knots[0], old.knots[-1], self.comparison.points)
        for curve, world, colour, label in ((old, self.scenario.world, "C0", "before"),
                                            (new, self.scenario.drifted, "C1", "after")):
            self._draw_curve(ax, curve, world, child, parent, reference, colour, label)
        changed = "truly changed" if (parent, child) in self.scenario.changed_edges else "not changed"
        ax.set_title(f"{parent} → {child}  ({changed}, gap {self.comparison.gap(old, new):.1f})", fontsize=9)
        ax.set_xlabel(parent)
        ax.axhline(0.0, color="grey", linewidth=0.5)

    def _draw_curve(self, ax, curve: Spline, world: World, child: str, parent: str,
                    reference: np.ndarray, colour: str, label: str) -> None:
        x = np.linspace(curve.knots[0], curve.knots[-1], self.comparison.points)
        values, spreads = self.comparison.levelled(curve, x, reference)
        ax.fill_between(x, values - 1.96 * spreads, values + 1.96 * spreads,
                        color=colour, alpha=0.2, linewidth=0)
        ax.plot(x, values, color=colour, label=label)
        truth = world.true_curve(child, parent)
        if truth is not None:
            ax.plot(x, truth(x) - truth(reference).mean(), color=colour, linestyle="--",
                    linewidth=1, label=f"truth {label}")


@dataclass
class EdgeResult:
    parent: str
    child: str
    truly_changed: bool
    gap: float
    flagged: bool


class DriftExperiment:
    """Fits one KAN on people from before the drift and one on people from after,
    then compares them edge by edge and saves the figures."""

    def __init__(self, scenario: DriftScenario, fitting: Fitting, intervals: int = 16,
                 size: int = 10_000, threshold: float = 3.0, seed: int = 0):
        self.scenario = scenario
        self.fitting = fitting
        self.intervals = intervals
        self.size = size
        self.threshold = threshold  # a gap above this many combined spreads counts as "changed"
        self.seed = seed
        self.comparison = EdgeComparison()

    def run(self, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        old_people = self.scenario.world.sample(self.size, self.seed)
        new_people = self.scenario.drifted.sample(self.size, self.seed + 1)
        before = self._fitted_model(self.scenario.world, old_people)
        after = self._fitted_model(self.scenario.drifted, new_people)
        self._save(KANPlot(before).figure(), directory / "before.png")
        self._save(KANPlot(after).figure(), directory / "after.png")
        self._save(DriftPlot(self.scenario, before, after, self.comparison).figure(), directory / "comparison.png")
        summary = self._summary(self._compare(before, after), before, after, old_people, new_people)
        (directory / "summary.txt").write_text(summary)
        return summary

    def _fitted_model(self, world: World, people: dict) -> CausalKAN:
        nodes = []
        for name, parents in world.parents().items():
            kind = OutcomeNode if name == world.outcome else Node
            nodes.append(kind(name, parents, self.intervals, fitting=self.fitting))
        model = CausalKAN(nodes)
        model.fit(people)
        return model

    def _compare(self, before: CausalKAN, after: CausalKAN) -> list[EdgeResult]:
        results = []
        for parent, child in self.scenario.world.edges():
            gap = self.comparison.gap(before.node(child).curves[parent], after.node(child).curves[parent])
            results.append(EdgeResult(parent, child, (parent, child) in self.scenario.changed_edges,
                                      gap, gap > self.threshold))
        return results

    def _summary(self, results: list[EdgeResult], before: CausalKAN, after: CausalKAN,
                 old_people: dict, new_people: dict) -> str:
        outcome = self.scenario.world.outcome
        lines = [f"World: {self.scenario.world.name}",
                 f"Drift: {self.scenario.description}",
                 "",
                 f"{'edge':22} {'truly changed':>14} {'gap (spreads)':>14} {'flagged (gap > ' + str(self.threshold) + ')':>20}"]
        for r in results:
            lines.append(f"{r.parent + ' → ' + r.child:22} {'yes' if r.truly_changed else 'no':>14} "
                         f"{r.gap:14.1f} {'yes' if r.flagged else 'no':>20}")
        lines += ["",
                  "Error in P(Y=1) against the true chance, averaged over people:",
                  f"  old model, people from before the drift  {self._chance_error(before, self.scenario.world, old_people):.3f}",
                  f"  old model, people from after the drift   {self._chance_error(before, self.scenario.drifted, new_people):.3f}",
                  f"  new model, people from after the drift   {self._chance_error(after, self.scenario.drifted, new_people):.3f}",
                  ""]
        return "\n".join(lines)

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
        experiment = DriftExperiment(scenario, BayesianFitting(), intervals=16)
        print(experiment.run(FIGURES / scenario.world.name))
    print(f"figures saved in {FIGURES}")


if __name__ == "__main__":
    main()
