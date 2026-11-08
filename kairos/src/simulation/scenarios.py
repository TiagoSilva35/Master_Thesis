"""The worlds to simulate, each with one drift.

Each scenario covers a different kind of drift:
  1. an equation between two features changes;
  2. the equation for Y changes (the bank changes its rule);
  3. only who the people are changes, no equation does;
  4. the same as 3, but Y has an income × savings term the KAN cannot represent.
"""

from dataclasses import dataclass, field

import numpy as np

from src.simulation.world import Equation, Outcome, Uniform, World


@dataclass
class DriftScenario:
    world: World                 # before the drift
    drifted: World               # after the drift
    description: str
    changed_edges: set[tuple[str, str]] = field(default_factory=set)  # (parent, child) whose equation truly changed


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
                         "but Y has a 0.15·income·savings term the KAN cannot represent")


SCENARIOS = [savings_habits, bank_rule, ageing_population, rising_savings_with_interaction]
