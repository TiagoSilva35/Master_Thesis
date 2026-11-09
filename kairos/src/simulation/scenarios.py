"""The worlds to simulate, each with one drift.

Each scenario covers a different kind of drift:
  1. an equation between two features changes;
  2. the equation for Y changes (the bank changes its rule);
  3. only who the people are changes, no equation does;
  4. the same as 3, but Y has an income × savings term a KAN of curves cannot represent;
  5. only the income × savings interaction changes (it flips sign);
  6. a larger world: five parents for Y, three interactions, two shifts in who
     the people are and one real change.

Which surfaces the KAN gets is decided by screening, in drift.py, not here.
"""

from dataclasses import dataclass, field

import numpy as np

from src.simulation.world import Equation, Outcome, Uniform, World


@dataclass
class DriftScenario:
    world: World                 # before the drift
    drifted: World               # after the drift
    description: str
    changed_edges: set[tuple[str, str]] = field(default_factory=set)  # (term, child) whose equation truly changed


def savings_habits() -> DriftScenario:
    world = World("1_savings_habits", {
        "income": Uniform(0.0, 4.0),
        "savings": Equation(0.0, {"income": lambda x: 0.5 * x**2}, noise=0.3),
        "Y": Outcome(-4.0, {"income": lambda x: x, "savings": lambda x: 0.5 * x}),
    })
    drifted = world.changed(
        savings=Equation(0.0, {"income": lambda x: 0.3 * x**2}, noise=0.3),
    )
    return DriftScenario(world, drifted,
                         "people save less: savings = 0.3·income² instead of 0.5·income²",
                         {("income", "savings")})


def bank_rule() -> DriftScenario:
    world = World("2_bank_rule", {
        "income": Uniform(0.0, 4.0),
        "savings": Equation(0.0, {"income": lambda x: 0.5 * x**2}, noise=0.3),
        "Y": Outcome(-3.0, {"income": lambda x: 2.0 * np.sin(1.5 * x), "savings": lambda x: 0.5 * x}),
    })
    drifted = world.changed(
        Y=Outcome(-3.0, {"income": lambda x: 1.5 * (x - 2.0), "savings": lambda x: 0.5 * x}),
    )
    return DriftScenario(world, drifted,
                         "the bank changes how income counts: 1.5·(income − 2) instead of 2·sin(1.5·income)",
                         {("income", "Y")})


def ageing_population() -> DriftScenario:
    world = World("3_ageing_population", {
        "age": Uniform(20.0, 60.0),
        "income": Equation(4.0, {"age": lambda x: -0.004 * (x - 50.0) ** 2}, noise=0.3),
        "Y": Outcome(-3.0, {"age": lambda x: 0.02 * (x - 45.0), "income": lambda x: x}),
    })
    drifted = world.changed(age=Uniform(35.0, 75.0))
    return DriftScenario(world, drifted,
                         "the population gets older: ages 35–75 instead of 20–60; no equation changes")


def rising_savings_with_interaction() -> DriftScenario:
    world = World("4_rising_savings_interaction", {
        "income": Uniform(0.0, 4.0),
        "savings": Uniform(0.0, 8.0),
        "Y": Outcome(-3.0, {"income": lambda x: 0.5 * x, "savings": lambda x: 0.25 * x},
                     interaction=lambda v: 0.15 * v["income"] * v["savings"]),
    })
    drifted = world.changed(savings=Uniform(4.0, 12.0))
    return DriftScenario(world, drifted,
                         "savings rise (4–12 instead of 0–8); no equation changes, "
                         "but Y has a 0.15·income·savings term a KAN of curves cannot represent")


def interaction_flips() -> DriftScenario:
    def world_with(strength: float) -> Outcome:
        return Outcome(-3.0, {"income": lambda x: 0.5 * x, "savings": lambda x: 0.25 * x},
                       interaction=lambda v: strength * (v["income"] - 2.0) * (v["savings"] - 4.0))

    world = World("5_interaction_flips", {
        "income": Uniform(0.0, 4.0),
        "savings": Uniform(0.0, 8.0),
        "Y": world_with(0.15),
    })
    drifted = world.changed(Y=world_with(-0.15))
    return DriftScenario(world, drifted,
                         "the interaction flips: −0.15·(income − 2)·(savings − 4) instead of +0.15·(…); "
                         "the curves' own effects stay the same",
                         {("income×savings", "Y")})


def large_world() -> DriftScenario:
    """Five variables, Y with five parents, three interactions, two shifts in who
    the people are and one real change.

    Interactions: age × education in income (strong), income × debt in Y
    (strong), age × savings in Y (weak). Drift: the population gets older and
    takes on more debt (no equation changes from that), and the bank becomes
    stricter on debt (the debt → Y edge truly changes).
    """
    def income_equation() -> Equation:
        return Equation(0.5, {"age": lambda x: 1.2 * np.sin(np.pi * (x - 20.0) / 60.0),
                              "education": lambda x: 0.15 * (x - 8.0)},
                        noise=0.3,
                        interaction=lambda v: 0.006 * (v["age"] - 45.0) * (v["education"] - 14.0))

    def outcome_equation(debt_strength: float) -> Outcome:
        return Outcome(-0.5, {"age": lambda x: 0.02 * (x - 45.0),
                              "education": lambda x: 0.0 * x,
                              "income": lambda x: 0.8 * (x - 2.0),
                              "savings": lambda x: 0.2 * (x - 3.0),
                              "debt": lambda x: debt_strength * (x - 2.5)},
                       interaction=lambda v: (0.3 * (v["income"] - 2.0) * (v["debt"] - 2.5)
                                              + 0.005 * (v["age"] - 45.0) * (v["savings"] - 3.0)))

    world = World("6_large_world", {
        "age": Uniform(20.0, 70.0),
        "education": Uniform(8.0, 20.0),
        "income": income_equation(),
        "savings": Equation(0.0, {"income": lambda x: 0.25 * (x + 1.0) ** 2,
                                  "age": lambda x: 0.02 * (x - 20.0)}, noise=0.4),
        "debt": Uniform(0.0, 5.0),
        "Y": outcome_equation(-0.6),
    })
    drifted = world.changed(age=Uniform(30.0, 80.0), debt=Uniform(1.0, 6.0), Y=outcome_equation(-1.0))
    return DriftScenario(world, drifted,
                         "older people (30–80) with more debt (1–6), and the bank is stricter on debt "
                         "(−1.0 instead of −0.6 per unit); interactions age×education, income×debt, age×savings",
                         {("debt", "Y")})


SCENARIOS = [savings_habits, bank_rule, ageing_population, rising_savings_with_interaction,
             interaction_flips, large_world]
