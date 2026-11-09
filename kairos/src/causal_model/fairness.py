"""Fairness: shape how a protected attribute may influence the decision."""

from dataclasses import dataclass, field

import numpy as np

from src.causal_model.kan import CausalKAN


@dataclass
class FairnessPolicy:
    """Which routes a protected attribute may influence the decision through.

    protected: the protected attribute, chosen beforehand (e.g. "sex").
    reference: the value it takes on routes where its influence is not allowed,
               for example the reference group 0.
    allowed:   for each child of the protected attribute (each edge out of it),
               the share of its influence allowed along that edge: 0 blocks it,
               1 keeps it, anything in between keeps that share. Edges that are
               not listed are kept.

    For example, FairnessPolicy("sex", 0, {"education": 1, "income": 0, "Y": 0})
    lets sex act on the decision through education, but not through pay or directly.
    """

    protected: str
    reference: float
    allowed: dict[str, float] = field(default_factory=dict)

    def seen_by(self, child: str, actual: np.ndarray) -> np.ndarray:
        """The protected attribute's value as the child's equation sees it."""
        share = self.allowed.get(child, 1.0)
        return self.reference + share * (np.asarray(actual, dtype=float) - self.reference)


class FairScore:
    """The decision's chance of a favourable outcome, with the protected
    attribute's influence shaped by a policy.

    The world model is not changed: it must stay true for counterfactuals and
    recourse. Leaving the protected attribute out of the model would not work
    either, because the variables it affects (income, education) would carry
    its influence instead. Only the score is shaped. For each person:
      1. recover each variable's noise (what makes this person who they are),
         as in a counterfactual;
      2. recompute every variable in causal order, keeping that noise, but
         every equation with the protected attribute as a parent sees it
         through the policy;
      3. the outcome's chance in that recomputed world is the fair score.
    Influence through a blocked edge is removed downstream too: if sex → income
    is blocked, income is recomputed as if for the reference group, and that
    income flows on to savings and Y. This is an edge-specific counterfactual
    (Avin, Shpitser & Pearl 2005; Chiappa 2019).
    """

    def __init__(self, model: CausalKAN, policy: FairnessPolicy, outcome: str = "Y"):
        self.model = model
        self.policy = policy
        self.outcome = outcome
        model.node(policy.protected)  # raises ValueError if the model has no such variable

    def of(self, people: dict[str, np.ndarray]) -> np.ndarray:
        return self.shaped_world(people)[self.outcome]

    def shaped_world(self, people: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Every variable recomputed with the protected attribute seen through the policy."""
        noise = {node.name: node.noise(people) for node in self.model.nodes}
        shaped: dict[str, np.ndarray] = {}
        for node in self.model.nodes:
            if not node.parents:
                shaped[node.name] = np.asarray(people[node.name], dtype=float)
                continue
            inputs = {parent: shaped[parent] for parent in node.parents}
            if self.policy.protected in node.parents:
                inputs[self.policy.protected] = self.policy.seen_by(node.name, people[self.policy.protected])
            shaped[node.name] = node.predict(inputs) + noise[node.name]
        return shaped

    def by_group(self, people: dict[str, np.ndarray]) -> dict[float, tuple[float, float]]:
        """For each value of the protected attribute: the average actual chance
        and the average fair chance."""
        actual = np.asarray(self.model.node(self.outcome).predict(people))
        fair = self.of(people)
        groups = np.asarray(people[self.policy.protected])
        return {float(g): (float(actual[groups == g].mean()), float(fair[groups == g].mean()))
                for g in np.unique(groups)}
