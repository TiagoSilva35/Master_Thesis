"""Did a node's equation change between two windows? One joint fit decides."""

import copy
from dataclasses import dataclass

import numpy as np

from src.causal_model.nodes import BayesianFitting, Fit, Node, PairScreening, SmoothnessBlock, Solve
from src.causal_model.terms import Levels, Spline

OFF = 1e8    # a strength so large that a change part is forced flat, i.e. switched off
START = 1e4  # where a change part's strength starts: almost off, so a change must earn its way in


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
    def __init__(self, margin: float = 3.0):
        self.margin = margin  # a change counts if the data becomes more than e^margin (≈ 20) times more probable
        
    def fit(self, node: Node, old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> ChangeResult:
        fitting = node.fitting
        if node.screening is not None:
            node.pairs = self._screened_pairs(node, node.screening, old, new)
        variables = [node.name] + node.parents
        pooled = {v: np.concatenate([old[v], new[v]]) for v in variables}
        after = np.concatenate([np.zeros(len(old[node.name])), np.ones(len(new[node.name]))])
        node.place_knots(pooled)
        shared = node.design(pooled)
        lift, change_columns, flattening = self._change_parts(node, pooled)
        size = shared.shape[1]
        design = np.hstack([shared, (shared @ lift) * after[:, None]])
        blocks = node.smoothness_blocks() + self._change_blocks(size, change_columns, flattening)
        solve = node.solver(design, node.target(pooled))
        result = fitting.fit(solve, design.shape[1], blocks)
        gains = self._gains(fitting, solve, result, blocks, size, design.shape[1])
        before, after_node = self._split(node, result, lift, size, pooled)
        return ChangeResult(before, after_node, gains, self.margin)

    @staticmethod
    def _screened_pairs(node: Node, screening: PairScreening, old: dict, new: dict) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for window in (old, new):
            for pair in screening.choose(node, window):
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
                assert isinstance(term, (Spline, Levels)), "curves must be Spline or Levels objects"
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
    def _change_blocks(size: int, change_columns: dict, flattening: dict) -> list[SmoothnessBlock]:
        level = SmoothnessBlock("change level", slice(size, size + 1), np.ones((1, 1)), start=START)
        return [level] + [SmoothnessBlock(f"change {label}", slice(size + c.start, size + c.stop), flattening[label], start=START)
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
                             result.evidence, smoothness=smoothness), pooled)
        return before, after
