"""Did a node's equation change between two windows? One joint fit decides."""

import copy
from dataclasses import dataclass

import numpy as np

from src.causal_model.spline import Spline
from src.causal_model.fitting import BayesianFitting, Fit, SmoothnessBlock, Solve
from src.causal_model.node import Node

OFF = 1e8  # a strength so large that a change part is forced flat, i.e. switched off


@dataclass
class ChangeResult:
    before: Node             # the node's equation before the drift (the shared parts)
    after: Node              # after the drift (shared plus change parts)
    gains: dict[str, float]  # per term, and "level": how much more probable the data is with its change on (log)
    margin: float

    def changed(self) -> list[str]:
        """The terms whose change made the data more than e^margin times more probable."""
        return [label for label, gain in self.gains.items() if gain > self.margin]


class ChangeFit:
    """Fits one node on people from before and after a drift together:

        x = shared terms + [person is from after the drift] × change terms + noise

    Every curve and surface gets a shared part, fitted on everyone, and a change
    part that applies only to people from after the drift. There is also a
    change in level (the constant). A change part is pulled towards flat and
    starts switched off; the fitting turns it on only if that makes the data
    more probable. Then each change is switched off again, one at a time, to
    measure how much the data needed it (its gain).

    Because there is one fit, a difference that changes no prediction (two
    correlated parents swapping credit) never makes the data more probable, so
    it is never counted; and both windows share one smoothing choice.

    If the node has a screening, each window chooses its surfaces and the node
    gets every pair either window chose (screening the two windows mixed
    together could miss an interaction that flips sign between them).

    The node's fitting must be a BayesianFitting: a PlainFitting ignores the
    beliefs that keep change parts switched off.
    """

    def __init__(self, margin: float = 3.0):
        self.margin = margin  # a change counts if the data becomes more than e^margin (≈ 20) times more probable
        
    def fit(self, node: Node, old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> ChangeResult:
        fitting = self._bayesian_fitting(node)
        if node.screening is not None:
            node.pairs = self._screened_pairs(node, old, new)
        variables = [node.name] + node.parents
        pooled = {v: np.concatenate([old[v], new[v]]) for v in variables}
        after = np.concatenate([np.zeros(len(old[node.name])), np.ones(len(new[node.name]))])
        node.place_knots(pooled)
        shared = node.design(pooled)
        lift, change_columns, flattening = self._change_parts(node, pooled)
        size = shared.shape[1]
        design = np.hstack([shared, (shared @ lift) * after[:, None]])
        blocks = node.smoothness_blocks() + self._change_blocks(fitting, size, change_columns, flattening)
        solve = node.solver(design, pooled[node.name])
        result = fitting.fit(solve, design.shape[1], blocks)
        gains = self._gains(fitting, solve, result, blocks, size, design.shape[1])
        before, after_node = self._split(node, result, lift, size, pooled)
        return ChangeResult(before, after_node, gains, self.margin)

    @staticmethod
    def _bayesian_fitting(node: Node) -> BayesianFitting:
        """The node's fitting, which must be a BayesianFitting: change parts are kept
        switched off by smoothness beliefs, which only a BayesianFitting has; with
        any other fitting every change would be free."""
        if not isinstance(node.fitting, BayesianFitting):
            raise TypeError(f"ChangeFit needs node '{node.name}' to use BayesianFitting, "
                            f"not {type(node.fitting).__name__}")
        return node.fitting

    @staticmethod
    def _screened_pairs(node: Node, old: dict, new: dict) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for window in (old, new):
            if node.screening is None:
                raise ValueError("node has no screening to choose pairs from")
            for pair in node.screening.choose(node, window):
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    @staticmethod
    def _change_parts(node: Node, pooled: dict) -> tuple[np.ndarray, dict[str, slice], dict[str, np.ndarray]]:
        """How change numbers turn into the node's own numbers (`lift`), which change
        numbers belong to each term, and each term's flattening cost.

        A curve's change is only allowed to average zero over the data, so a
        change in level shows up only in the constant's change, never in a curve.
        """
        columns = node.columns()
        size = 1 + sum(c.stop - c.start for c in columns.values())
        bases, change_columns, flattening, start = {}, {}, {}, 1
        for label, term in node.terms().items():
            if label in node.curves:
                assert isinstance(term, Spline), "curves must be Spline objects"
                basis = term.zero_average_basis(pooled[label])
                flattening[label] = basis.T @ term.flattening() @ basis
            else:
                basis = np.eye(columns[label].stop - columns[label].start)  # surfaces already average zero
                flattening[label] = term.flattening()
            bases[label] = basis
            change_columns[label] = slice(start, start + basis.shape[1])
            start += basis.shape[1]
        lift = np.zeros((size, start))
        lift[0, 0] = 1.0  # the constant's change
        for label, basis in bases.items():
            lift[columns[label], change_columns[label]] = basis
        return lift, change_columns, flattening

    @staticmethod
    def _change_blocks(fitting: BayesianFitting, size: int, change_columns: dict, flattening: dict) -> list[SmoothnessBlock]:
        top = max(fitting.grid)
        level = SmoothnessBlock("change level", slice(size, size + 1), np.ones((1, 1)), start=top)
        return [level] + [SmoothnessBlock(f"change {label}", slice(size + c.start, size + c.stop), flattening[label], start=top)
                          for label, c in change_columns.items()]

    @staticmethod
    def _gains(fitting: BayesianFitting, solve: Solve, result: Fit, blocks: list[SmoothnessBlock], size: int, total: int) -> dict[str, float]:
        """For each change, the level change included: how much more probable the
        data is with it on than with it switched off (in log units)."""
        gains = {}
        for index, block in enumerate(blocks):
            if block.label.startswith("change "):
                strengths = list(result.smoothness)
                strengths[index] = OFF
                off = solve(fitting.penalty(strengths, total, blocks))
                gains[block.label.removeprefix("change ")] = result.evidence - off.evidence
        return gains

    @staticmethod
    def _split(node: Node, result: Fit, lift: np.ndarray, size: int, pooled: dict) -> tuple[Node, Node]:
        """Two ordinary nodes from the joint fit: before (shared parts) and after (shared + change)."""
        smoothness = list(result.smoothness[:len(node.smoothness_blocks())])
        before, after = copy.deepcopy(node), copy.deepcopy(node)
        keep_shared = np.hstack([np.eye(size), np.zeros((size, lift.shape[1]))])
        add_change = np.hstack([np.eye(size), lift])
        for target, combine in ((before, keep_shared), (after, add_change)):
            target.store(Fit(combine @ result.coefficients, combine @ result.covariance @ combine.T,
                             result.evidence, smoothness), pooled)
        return before, after
