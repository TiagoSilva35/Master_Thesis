import math

import matplotlib.pyplot as plt
import numpy as np

from src.causal_model.kan import CausalKAN
from src.causal_model.outcome import OutcomeNode


class KANPlot:
    # One panel per edge: the parent on x, the curve (its push on the child) on y.

    def __init__(self, model: CausalKAN, points: int = 200, columns: int = 3):
        self.model = model
        self.points = points
        self.columns = columns

    def figure(self):
        edges = [(parent, node) for node in self.model.nodes for parent in node.parents]
        if not edges:
            raise ValueError("the model has no edges to plot")
        columns = min(self.columns, len(edges))
        rows = math.ceil(len(edges) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for ax, (parent, node) in zip(axes.flat, edges):
            self._draw_edge(ax, parent, node)
        for ax in axes.flat[len(edges):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig

    def _draw_edge(self, ax, parent: str, node) -> None:
        curve = node.curves[parent]
        x = np.linspace(curve.knots[0], curve.knots[-1], self.points)
        if curve.covariance is not None:
            low, high = curve.band(x)
            ax.fill_between(x, low, high, alpha=0.25, linewidth=0)   # 95% band
        ax.plot(x, curve(x))
        ax.plot(curve.knots, curve(curve.knots), "o", markersize=3)  # the knots
        ax.axhline(0.0, color="grey", linewidth=0.5)                  # the node's average
        title = f"{parent} → {node.name}"
        if parent in node.smoothness:
            title += f"  (λ = {node.smoothness[parent]:g})"
        ax.set_title(title)
        ax.set_xlabel(parent)
        unit = "log-odds" if isinstance(node, OutcomeNode) else node.name
        ax.set_ylabel(f"push on {node.name} ({unit})")
