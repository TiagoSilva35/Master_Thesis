"""KaCGM's ingredients turned into a discovery method (KaCGM itself takes the graph as given).

KaCGM (Almodóvar et al. 2026, arXiv 2603.20184) writes each variable as a KAN of its
parents plus noise (a logistic, ordered or softmax KAN for a discrete one) and checks a
given graph by testing, with HSIC, whether the recovered noise is independent of the
parents. Here that check drives a search, as in RESIT (Peters et al. 2014):

1. Order: among the variables left, the one whose noise depends least on all the others
   is the last one. Remove it and repeat.
2. Parents: each variable starts with everything before it, and loses the parents whose
   removal costs less than the price per arrow (the same evidence and price as kairos).

The node is KaCGM's KAAM (one curve per parent, no hidden layer) written with kairos'
splines, so this is a stand-in, not their code. For a discrete variable the noise is
KaCGM's inverse-CDF one: a uniform draw between P(x < value) and P(x ≤ value).
"""
import numpy as np

from src.causal_model.discovery import Background, NodeMaker
from src.causal_model.nodes import CategoryNode, OrdinalNode, BinaryNode


class Resit:
    def __init__(self, background: Background, price: float = 4.0, seed: int = 0, surfaces: bool = False):
        self.surfaces = surfaces  # two surfaces per node, for what a KAN's hidden layer could add
        self.maker = NodeMaker(background)
        self.background = background
        self.price = price
        self.random = np.random.default_rng(seed)
        self.kernels: dict = {}

    def arrows(self, data: dict) -> set:
        data = {name: values for name, values in data.items() if np.ptp(values) > 0}
        self.kernels = {name: self._kernel(self._numbers(name, values)) for name, values in data.items()}
        left, parents = sorted(data), {}
        while len(left) > 1:
            dependence = {name: self._dependence(name, [other for other in left if other != name], data) for name in left
                          if name not in self.background.roots or all(other in self.background.roots for other in left)}
            last = min(dependence, key=lambda name: dependence[name])
            left.remove(last)
            parents[last] = self._needed(last, list(left), data)
        return {(parent, child) for child, ps in parents.items() for parent in ps}

    def _node(self, name, parents):
        node = self.maker.make(name, parents)
        if not self.surfaces:
            node.screening = None
        node.read_noise = lambda fit, data: None
        return node

    def _evidence(self, name, parents, data):
        node = self._node(name, parents)
        node.fit(data)
        return node.evidence

    def _needed(self, name, parents, data):
        if name in self.background.roots:
            return []
        current = self._evidence(name, parents, data)
        while parents:
            trials = {parent: self._evidence(name, [p for p in parents if p != parent], data) for parent in parents}
            weakest = max(trials, key=lambda parent: trials[parent])
            if current - trials[weakest] >= self.price:
                return parents
            parents, current = [p for p in parents if p != weakest], trials[weakest]
        return parents

    def _dependence(self, name, others, data):
        """HSIC between the noise and all the others, as a share of its largest value (0 = independent)."""
        node = self._node(name, others)
        node.fit(data)
        noise = self._kernel(self._noise(node, data)[:, None])
        joint = np.ones_like(noise)
        for other in others:
            joint = joint * self.kernels[other] ** (1 / len(others))  # a Gaussian kernel on the average squared distance
        noise, joint = self._centred(noise), self._centred(joint)
        return float(np.sum(noise * joint) / np.sqrt(np.sum(noise ** 2) * np.sum(joint ** 2)))

    def _noise(self, node, data):
        values = data[node.name]
        if isinstance(node, BinaryNode):
            chances = np.column_stack([1 - node.predict(data), node.predict(data)])
            codes = (values == values.max()).astype(int)
        elif isinstance(node, (OrdinalNode, CategoryNode)):
            chances, codes = node.probabilities(data), np.searchsorted(node.levels, values)
        else:
            return node.noise(data)
        upto = np.cumsum(chances, axis=1)
        rows = np.arange(len(values))
        low = np.where(codes > 0, upto[rows, codes - 1], 0.0)
        return low + self.random.uniform(size=len(values)) * (upto[rows, codes] - low)

    def _numbers(self, name, values):
        if self.background.kind(name) == "nominal":
            return (values[:, None] == np.unique(values)).astype(float)
        return values[:, None].astype(float)

    @staticmethod
    def _kernel(x):
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
        squares = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
        middle = np.median(squares[squares > 0]) if np.any(squares > 0) else 1.0
        return np.exp(-squares / middle)

    @staticmethod
    def _centred(kernel):
        return kernel - kernel.mean(0)[None, :] - kernel.mean(1)[:, None] + kernel.mean()
