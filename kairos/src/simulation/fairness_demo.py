"""Fairness demo: a world with a protected attribute, a KAN fitted to it, and fair scores.

The world: group (0 or 1) is the protected attribute. Group 1 gets a little less
education, is paid less for the same education, is treated worse directly by the
bank, and its income counts for less (a group × income interaction).

Each policy says which routes group may still act through; the KAN's fair score
is checked against the same computation done with the world's true equations.
"Unaware" is the common shortcut of leaving group out of the model.

Saves figures/fairness/{kan.png, scores.png, summary.txt}.
Run from kairos/:  PYTHONPATH=. venv/bin/python src/simulation/fairness_demo.py
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.causal_model.fairness import FairnessPolicy, FairScore
from src.causal_model.fitting import BayesianFitting
from src.causal_model.kan import CausalKAN
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.causal_model.plot import KANPlot
from src.causal_model.screening import PairScreening
from src.simulation.world import Binary, Equation, Outcome, World

FIGURES = Path(__file__).resolve().parents[2] / "figures" / "fairness"

POLICIES = {
    "block pay gap and direct": FairnessPolicy("group", 0.0, {"education": 1.0, "income": 0.0, "Y": 0.0}),
    "block every route": FairnessPolicy("group", 0.0, {"education": 0.0, "income": 0.0, "Y": 0.0}),
    "half the pay gap, no direct": FairnessPolicy("group", 0.0, {"education": 1.0, "income": 0.5, "Y": 0.0}),
}


def fairness_world() -> World:
    return World("fairness", {
        "group": Binary(0.5),
        "education": Equation(14.0, {"group": lambda g: -1.0 * g}, noise=2.0),
        "income": Equation(1.0, {"education": lambda e: 0.2 * (e - 14.0), "group": lambda g: -0.6 * g}, noise=0.4),
        "Y": Outcome(-0.3, {"group": lambda g: -0.5 * g, "income": lambda x: 0.9 * (x - 1.0)},
                     interaction=lambda v: -0.4 * v["group"] * (v["income"] - 1.0)),
    })


class TrueFairScore:
    """The same edge-specific counterfactual as FairScore, but with the world's
    true equations, to check the KAN against."""

    def __init__(self, world: World, policy: FairnessPolicy):
        self.world = world
        self.policy = policy

    def of(self, people: dict[str, np.ndarray]) -> np.ndarray:
        shaped: dict[str, np.ndarray] = {}
        for name, mechanism in self.world.mechanisms.items():
            if not isinstance(mechanism, Equation) or not mechanism.edges:
                shaped[name] = people[name]  # no parents: nothing to shape
                continue
            inputs = {parent: shaped[parent] for parent in mechanism.edges}
            if self.policy.protected in mechanism.edges:
                inputs[self.policy.protected] = self.policy.seen_by(name, people[self.policy.protected])
            if isinstance(mechanism, Outcome):
                shaped[name] = mechanism.chance(inputs)
            else:
                shaped[name] = mechanism.mean(inputs) + (people[name] - mechanism.mean(people))
        return shaped[self.world.outcome]


class FairnessDemo:
    def __init__(self, size: int = 10_000, seed: int = 0):
        self.world = fairness_world()
        self.people = self.world.sample(size, seed)
        self.model = self._fitted_model()

    def run(self) -> str:
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig = KANPlot(self.model).figure()
        fig.savefig(FIGURES / "kan.png", dpi=110)
        plt.close(fig)
        rows = self._rows()
        self._scores_figure(rows)
        text = self._summary(rows)
        (FIGURES / "summary.txt").write_text(text)
        return text

    def _fitted_model(self) -> CausalKAN:
        fitting = BayesianFitting()
        model = CausalKAN([
            Node("group", [], pairs=[], intervals=16, fitting=fitting),
            Node("education", ["group"], pairs=[], intervals=16, fitting=fitting),
            Node("income", ["education", "group"], pairs=[], intervals=16, fitting=fitting),
            OutcomeNode("Y", ["group", "income"], pairs=[], intervals=16, fitting=fitting,
                        screening=PairScreening()),
        ])
        model.fit(self.people)
        return model

    def _rows(self) -> dict[str, dict]:
        """For the actual decision, each policy, and "unaware": average chance per
        group from the KAN and from the truth, and the KAN's error per person."""
        group = self.people["group"]
        true_chance = self.world.chance(self.people)
        model_chance = np.asarray(self.model.node("Y").predict(self.people))
        rows = {"actual decision": self._row(model_chance, true_chance, group)}
        for name, policy in POLICIES.items():
            rows[name] = self._row(FairScore(self.model, policy).of(self.people),
                                   TrueFairScore(self.world, policy).of(self.people), group)
        unaware = OutcomeNode("Y", ["income"], pairs=[], intervals=16, fitting=BayesianFitting())
        unaware.fit(self.people)
        rows["unaware (group left out)"] = self._row(unaware.predict(self.people), None, group)
        return rows

    @staticmethod
    def _row(model: np.ndarray, truth: np.ndarray | None, group: np.ndarray) -> dict:
        return {"model": {g: float(model[group == g].mean()) for g in (0.0, 1.0)},
                "truth": None if truth is None else {g: float(truth[group == g].mean()) for g in (0.0, 1.0)},
                "error": None if truth is None else float(np.mean(np.abs(model - truth))),
                "values": model, "true values": truth}

    def _scores_figure(self, rows: dict) -> None:
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
        names = list(rows)
        x = np.arange(len(names))
        for offset, g, colour in ((-0.2, 0.0, "C0"), (0.2, 1.0, "C3")):
            left.bar(x + offset, [rows[n]["model"][g] for n in names], width=0.38, color=colour,
                     label=f"group {int(g)} (KAN)")
            truths = [rows[n]["truth"][g] if rows[n]["truth"] else np.nan for n in names]
            left.plot(x + offset, truths, "k_", markersize=14, label="truth" if g == 0.0 else None)
        left.set_xticks(x, [textwrap.fill(n, 14) for n in names], fontsize=8)
        left.set_ylabel("average chance of a favourable outcome")
        left.legend(fontsize=8)
        left.set_title("Average chance per group")
        chosen = rows["block pay gap and direct"]
        group = self.people["group"]
        for g, colour in ((0.0, "C0"), (1.0, "C3")):
            pick = group == g
            right.scatter(chosen["true values"][pick][:1500], chosen["values"][pick][:1500], s=3,
                          color=colour, alpha=0.4, label=f"group {int(g)}")
        right.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
        right.set_xlabel("true fair score")
        right.set_ylabel("KAN fair score")
        right.set_title("Policy 'block pay gap and direct': KAN vs truth, per person")
        right.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "scores.png", dpi=110)
        plt.close(fig)

    def _summary(self, rows: dict) -> str:
        surfaces = self.model.node("Y").pairs or "none"
        lines = ["Fairness demo: protected attribute 'group' (0 or 1), 10,000 people.",
                 f"Surfaces chosen for Y by screening: {surfaces}", "",
                 f"{'':30} {'group 0':>16} {'group 1':>16} {'gap':>12} {'KAN error':>10}",
                 f"{'':30} {'KAN (truth)':>16} {'KAN (truth)':>16} {'KAN (truth)':>12} {'per person':>10}"]
        for name, row in rows.items():
            m, t = row["model"], row["truth"]
            cells = [f"{m[g]:.3f}" + (f" ({t[g]:.3f})" if t else "") for g in (0.0, 1.0)]
            gap = f"{m[0.0] - m[1.0]:.3f}" + (f" ({t[0.0] - t[1.0]:.3f})" if t else "")
            error = f"{row['error']:.3f}" if row["error"] is not None else "–"
            lines.append(f"{name:30} {cells[0]:>16} {cells[1]:>16} {gap:>12} {error:>10}")
        lines += ["", "Policies (share of group's influence allowed along each edge out of group):"]
        lines += [f"  {name}: {policy.allowed}" for name, policy in POLICIES.items()]
        return "\n".join(lines) + "\n"


def main() -> None:
    print(FairnessDemo().run())
    print(f"figures saved in {FIGURES}")


if __name__ == "__main__":
    main()
