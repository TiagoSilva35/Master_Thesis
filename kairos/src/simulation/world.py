"""A true world we can sample people from, with its equations known."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Uniform:
    """A variable with no parents, spread evenly between low and high."""

    low: float
    high: float

    def draw(self, values: dict, size: int, random: np.random.Generator) -> np.ndarray:
        return random.uniform(self.low, self.high, size)


@dataclass
class Equation:
    """A variable with parents: constant + Σ edge(parent) + interaction + bell-shaped noise.

    `edges` holds one function per parent, like the KAN's curves. `interaction`
    is an optional term that mixes parents (for example income × savings),
    which a sum of one-parent curves cannot represent.
    """

    constant: float
    edges: dict[str, Callable[[np.ndarray], np.ndarray]]
    noise: float = 0.0
    interaction: Callable[[dict], np.ndarray] | None = None

    def mean(self, values: dict) -> np.ndarray:
        total = self.constant + sum(edge(values[parent]) for parent, edge in self.edges.items())
        if self.interaction is not None:
            total = total + self.interaction(values)
        return total

    def draw(self, values: dict, size: int, random: np.random.Generator) -> np.ndarray:
        return self.mean(values) + random.normal(0.0, self.noise, size)


@dataclass
class Outcome(Equation):
    """The outcome Y: 1 with chance σ(constant + Σ edge(parent) + interaction)."""

    def chance(self, values: dict) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.mean(values)))

    def draw(self, values: dict, size: int, random: np.random.Generator) -> np.ndarray:
        return (random.uniform(size=size) < self.chance(values)).astype(float)


@dataclass
class World:
    """A true causal world: one mechanism per variable, written in causal order."""

    name: str
    mechanisms: dict[str, Uniform | Equation]
    outcome: str = "Y"

    def sample(self, size: int, seed: int) -> dict[str, np.ndarray]:
        """Draw `size` people, each variable after its parents."""
        random = np.random.default_rng(seed)
        values: dict[str, np.ndarray] = {}
        for name, mechanism in self.mechanisms.items():
            values[name] = mechanism.draw(values, size, random)
        return values

    def changed(self, **mechanisms: Uniform | Equation) -> "World":
        """A copy of this world with some mechanisms replaced (the drift)."""
        return World(self.name, {**self.mechanisms, **mechanisms}, self.outcome)

    def parents(self) -> dict[str, list[str]]:
        return {name: list(getattr(mechanism, "edges", {})) for name, mechanism in self.mechanisms.items()}

    def edges(self) -> list[tuple[str, str]]:
        """Every (parent, child) pair."""
        return [(parent, child) for child, parents in self.parents().items() for parent in parents]

    def chance(self, values: dict) -> np.ndarray:
        """The true P(Y = 1) for these people."""
        return self.mechanisms[self.outcome].chance(values)

    def true_curve(self, child: str, parent: str) -> Callable | None:
        """The true edge function, or None when the child has an interaction
        (then no single curve per edge is the truth)."""
        mechanism = self.mechanisms[child]
        if mechanism.interaction is not None:
            return None
        return mechanism.edges[parent]
