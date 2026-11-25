import math

import matplotlib.pyplot as plt
import numpy as np

from src.causal_model.kan import CausalKAN
from src.causal_model.nodes import LaplaceNode
from src.causal_model.terms import Levels


class KANPlot:
    # One panel per edge: the parent on x, the curve (its push on the child) on y.
    # One heat map per surface: both parents on the axes, the colour is the push.

    def __init__(self, model: CausalKAN, points: int = 200, columns: int = 3):
        self.model = model
        self.points = points
        self.columns = columns

    def figure(self):
        panels = [(parent, node) for node in self.model.nodes for parent in node.parents]
        panels += [(pair, node) for node in self.model.nodes for pair in node.pairs]
        if not panels:
            raise ValueError("the model has no edges to plot")
        columns = min(self.columns, len(panels))
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False)
        for ax, (term, node) in zip(axes.flat, panels):
            if isinstance(term, tuple):
                self._draw_surface(fig, ax, term, node)
            else:
                self._draw_edge(ax, term, node)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig

    def _draw_surface(self, fig, ax, pair: tuple[str, str], node) -> None:
        surface = node.surfaces[pair]
        first, second = pair
        grid1, grid2 = np.meshgrid(self._axis(surface.first), self._axis(surface.second))
        values = surface(grid1.ravel(), grid2.ravel()).reshape(grid1.shape)
        limit = float(np.max(np.abs(values))) or 1.0
        mesh = ax.pcolormesh(grid1, grid2, values, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        fig.colorbar(mesh, ax=ax, label=f"push on {node.name}")
        title = f"{first} × {second} → {node.name}"
        label = node.surface_label(pair)
        if label in node.smoothness:
            title += f"  (λ = {node.smoothness[label]:.3g})"
        ax.set_title(title)
        ax.set_xlabel(first)
        ax.set_ylabel(second)

    @staticmethod
    def _axis(side) -> np.ndarray:
        """Where to draw one side of a surface: its levels, or 60 points along its range."""
        if isinstance(side, Levels):
            return side.levels
        return np.linspace(side.knots[0], side.knots[-1], 60)

    def _draw_edge(self, ax, parent: str, node) -> None:
        curve = node.curves[parent]
        if isinstance(curve, Levels):
            self._draw_levels(ax, curve)
        else:
            x = np.linspace(curve.knots[0], curve.knots[-1], self.points)
            if curve.covariance is not None:
                low, high = curve.band(x)
                ax.fill_between(x, low, high, alpha=0.25, linewidth=0)   # 95% band
            ax.plot(x, curve(x))
            ax.plot(curve.knots, curve(curve.knots), "o", markersize=3)  # the knots
        ax.axhline(0.0, color="grey", linewidth=0.5)                      # the node's average
        title = f"{parent} → {node.name}"
        if parent in node.smoothness:
            title += f"  (λ = {node.smoothness[parent]:.3g})"
        ax.set_title(title)
        ax.set_xlabel(parent)
        unit = "log-odds" if isinstance(node, LaplaceNode) else node.name
        if node.scale.name != "plain":
            unit = f"{node.scale.name} scale"
        ax.set_ylabel(f"push on {node.name} ({unit})")

    @staticmethod
    def _draw_levels(ax, levels: Levels) -> None:
        """One point per level, with its 95% band as an error bar."""
        x = levels.levels
        if levels.covariance is not None:
            low, high = levels.band(x)
            ax.errorbar(x, levels(x), yerr=[levels(x) - low, high - levels(x)], fmt="o", capsize=3)
        else:
            ax.plot(x, levels(x), "o")
        ax.set_xticks(x)
