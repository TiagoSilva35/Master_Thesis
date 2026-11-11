"""Draws the figures for the Fair Causal KAN deck into assets/.

Every figure illustrates a formula from the slides; the spline weights come from
kairos' own Spline class, so the pictures match the code.

Run from this folder:  ../../kairos/venv/bin/python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kairos"))
from src.causal_model.spline import Spline  # noqa: E402

ASSETS = HERE / "assets"

RED, INK, BODY, GREY, LGREY, AMBER, BLUE = ("#DA0725", "#141414", "#262626", "#8A8A8A",
                                           "#C4C4C4", "#C77A18", "#2F5D8A")
DIVERGING = LinearSegmentedColormap.from_list("blue_white_red", [BLUE, "#FFFFFF", RED])


class Style:
    """Matplotlib settings that match the deck: Helvetica, thin grey axes."""

    @staticmethod
    def apply() -> None:
        plt.rcParams.update({
            "font.family": ["Helvetica", "Arial"],
            "font.size": 13,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
            "axes.edgecolor": GREY,
            "axes.labelcolor": BODY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "xtick.labelcolor": BODY,
            "ytick.labelcolor": BODY,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        })


class BayesianSpline:
    """The Gaussian Bayesian P-spline of the slides, on one curve (no constant:
    the bump weights add up to 1, so the heights carry the level)."""

    RIDGE = 1e-3

    def __init__(self, spline: Spline, x: np.ndarray, y: np.ndarray):
        self.spline = spline
        self.design = spline.weights(x)
        self.y = y

    def penalty(self, strength: float) -> np.ndarray:
        size = len(self.spline.heights)
        return self.RIDGE * np.eye(size) + strength * self.spline.bending()

    def fit(self, strength: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Most believable heights, their covariance, and the evidence."""
        penalty = self.penalty(strength)
        precision = self.design.T @ self.design + penalty
        heights = np.linalg.solve(precision, self.design.T @ self.y)
        leftover = np.sum((self.y - self.design @ heights) ** 2) + heights @ penalty @ heights
        noise = leftover / len(self.y)
        evidence = (-0.5 * len(self.y) * np.log(noise) + 0.5 * np.linalg.slogdet(penalty)[1]
                    - 0.5 * np.linalg.slogdet(precision)[1])
        return heights, noise * np.linalg.inv(precision), float(evidence)

    def best(self, grid: np.ndarray) -> float:
        return float(max(grid, key=lambda strength: self.fit(strength)[2]))


class Figures:
    def __init__(self, seed: int = 4):
        self.random = np.random.default_rng(seed)

    def all(self) -> None:
        ASSETS.mkdir(exist_ok=True)
        self.bezier()
        self.bezier_weights()
        self.bspline()
        self.drift()
        self.prior_posterior()
        self.evidence()
        self.surface()
        self.fast()

    # ---------------------------------------------------------------- slide 1
    def bezier(self) -> None:
        points = np.array([[0.0, 0.0], [0.9, 2.6], [3.4, 2.9], [4.2, 0.5]])
        t = np.linspace(0, 1, 200)[:, None]
        curve = ((1 - t) ** 3 * points[0] + 3 * t * (1 - t) ** 2 * points[1]
                 + 3 * t ** 2 * (1 - t) * points[2] + t ** 3 * points[3])
        fig, ax = plt.subplots(figsize=(6.0, 4.3))
        ax.add_patch(Polygon(points, closed=True, facecolor="#F4D9DC", edgecolor="none", zorder=0))
        ax.plot(*points.T, color=GREY, linestyle="--", linewidth=1.2, zorder=1)
        ax.plot(*curve.T, color=RED, linewidth=3, zorder=2)
        half = curve[100]
        ax.plot(*half, "o", color=INK, markersize=7, zorder=4)
        ax.annotate("f(0.5) = ⅛V₁ + ⅜V₂ + ⅜V₃ + ⅛V₄", half, xytext=(0.95, 1.05), fontsize=12, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8))
        offsets = [(-0.32, -0.15), (-0.35, 0.08), (0.12, 0.08), (0.12, -0.15)]
        for (px, py), (dx, dy), label in zip(points, offsets, ["V₁", "V₂", "V₃", "V₄"]):
            ax.plot(px, py, "o", markersize=10, markerfacecolor="white", markeredgecolor=INK,
                    markeredgewidth=1.6, zorder=3)
            ax.text(px + dx, py + dy, label, fontsize=15, fontweight="bold", color=INK)
        ax.text(2.1, 2.45, "convex hull", fontsize=12, color=RED, ha="center", style="italic")
        ax.set_xlim(-0.6, 4.7)
        ax.set_ylim(-0.4, 3.3)
        ax.set_aspect("equal")
        ax.axis("off")
        self._save(fig, "bezier.pdf")

    def bezier_weights(self) -> None:
        t = np.linspace(0, 1, 200)
        weights = [(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3]
        fig, ax = plt.subplots(figsize=(4.6, 2.35))
        for w, colour, label in zip(weights, [INK, RED, AMBER, BLUE], ["V₁", "V₂", "V₃", "V₄"]):
            ax.plot(t, w, color=colour, linewidth=2.2, label=label)
        ax.axvline(0.5, color=LGREY, linewidth=1, linestyle=":")
        ax.set_xlabel("t")
        ax.set_ylabel("weight")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.legend(ncol=4, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.25), handlelength=1.2)
        self._save(fig, "bezier_weights.pdf")

    # ---------------------------------------------------------------- slide 2
    def bspline(self) -> None:
        spline = Spline(0.0, 3.0, 3)
        spline.heights = np.array([0.4, 1.6, 0.6, 1.8, 1.0, 0.5])
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True,
                                          gridspec_kw={"height_ratios": [1.7, 1]})
        for knot in spline.knots:
            for ax in (top, bottom):
                ax.axvline(knot, color=LGREY, linewidth=0.9, linestyle="--", zorder=0)
        top.plot(spline.centers, spline.heights, color=GREY, linestyle="--", linewidth=1.1)
        top.plot(spline.centers, spline.heights, "o", markersize=8, markerfacecolor="white",
                 markeredgecolor=INK, markeredgewidth=1.4)
        for piece, colour in zip(range(3), [RED, INK, AMBER]):
            x = np.linspace(spline.knots[piece], spline.knots[piece + 1], 80)
            top.plot(x, spline(x), color=colour, linewidth=3.2, solid_capstyle="butt")
        for knot in spline.knots[1:-1]:
            top.plot(knot, spline(knot), "o", color=INK, markersize=6, zorder=5)
        top.text(spline.centers[0], spline.heights[0] - 0.3, "height 1", fontsize=11, color=GREY, ha="center")
        top.text(spline.centers[-1], spline.heights[-1] - 0.3, "height 6", fontsize=11, color=GREY, ha="center")
        for ax in (top, bottom):
            ax.axvspan(-1.3, 0, color="#F5F5F5", zorder=0)
            ax.axvspan(3, 4.3, color="#F5F5F5", zorder=0)
        top.set_ylim(0, 2.1)
        top.set_yticks([])
        top.spines["left"].set_visible(False)
        x = np.linspace(spline.knots[0], spline.knots[-1], 300)
        weights = spline.weights(x)
        for k in range(len(spline.heights)):
            bottom.plot(x, weights[:, k], color=GREY, linewidth=1.6)
        point = 1.3
        for w in spline.weights(np.array([point]))[0]:
            if w > 0:
                bottom.plot(point, w, "o", color=RED, markersize=6)
        bottom.axvline(point, color=RED, linewidth=0.9)
        bottom.set_ylabel("bump weights")
        bottom.set_yticks([0, 1 / 6, 2 / 3], ["0", "1/6", "4/6"])
        bottom.set_xticks(spline.knots, ["knot 0", "knot 1", "knot 2", "knot 3"])
        bottom.set_xlim(-1.3, 4.3)
        self._save(fig, "bspline.pdf")

    # ---------------------------------------------------------------- slide 3
    def drift(self) -> None:
        def truth(x):
            return np.sin(1.6 * x) + 0.3 * x

        x = np.concatenate([self.random.uniform(0, 2.6, 220), self.random.uniform(2.6, 4, 7)])
        y = truth(x) + self.random.normal(0, 0.3, len(x))
        spline = Spline(0.0, 4.0, 20)
        model = BayesianSpline(spline, x, y)
        grid = 10 ** np.arange(-1, 4.5, 0.5)
        grid_x = np.linspace(0, 4, 300)
        weights = spline.weights(grid_x)
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.5), sharey=True)
        for ax, title in zip(axes, ["Plain B-spline: no smoothness belief (λ = 0)", "Bayesian spline: λ from the data, with a band"]):
            ax.axvspan(2.6, 4, color="#F5F5F5", zorder=0)
            ax.text(3.3, -1.55, "few people here yet", ha="center", fontsize=11, color=GREY)
            ax.plot(x, y, "o", color=LGREY, markersize=3.5, zorder=1)
            ax.plot(grid_x, truth(grid_x), color=INK, linestyle="--", linewidth=1.2, label="truth")
            ax.set_title(title, fontsize=13, color=INK, loc="left")
            ax.set_xlabel("income")
        plain, _, _ = model.fit(0.0)
        axes[0].plot(grid_x, weights @ plain, color=AMBER, linewidth=2.6, label="fit")
        heights, covariance, _ = model.fit(model.best(grid))
        middle = weights @ heights
        spread = np.sqrt(np.sum((weights @ covariance) * weights, axis=1))
        axes[1].fill_between(grid_x, middle - 1.96 * spread, middle + 1.96 * spread, color=RED, alpha=0.18,
                             linewidth=0, label="95% band")
        axes[1].plot(grid_x, middle, color=RED, linewidth=2.6, label="fit")
        axes[0].set_ylim(-1.9, 3.0)
        axes[0].set_ylabel("savings")
        for ax in axes:
            ax.legend(fontsize=11, loc="upper left")
        self._save(fig, "drift.pdf")

    # ---------------------------------------------------------------- slide 4
    def prior_posterior(self) -> None:
        spline = Spline(0.0, 4.0, 12)
        x = self.random.uniform(0, 4, 18)
        y = np.sin(1.4 * x) + self.random.normal(0, 0.25, len(x))
        model = BayesianSpline(spline, x, y)
        strength = 3.0
        prior_precision = strength * spline.bending() + 0.6 * np.eye(len(spline.heights))
        heights, covariance, _ = model.fit(strength)
        grid_x = np.linspace(0, 4, 300)
        weights = spline.weights(grid_x)
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
        prior = self.random.multivariate_normal(np.zeros(len(heights)), np.linalg.inv(prior_precision), 12)
        posterior = self.random.multivariate_normal(heights, covariance, 12)
        for draw in prior:
            axes[0].plot(grid_x, weights @ draw, color=GREY, linewidth=1.1, alpha=0.8)
        for draw in posterior:
            axes[1].plot(grid_x, weights @ draw, color=RED, linewidth=1.1, alpha=0.55)
        axes[1].plot(x, y, "o", color=INK, markersize=4.5, zorder=3)
        axes[0].set_title("before the data: smooth guesses", fontsize=12, color=INK, loc="left")
        axes[1].set_title("after the data: only those that fit", fontsize=12, color=INK, loc="left")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("parent")
        axes[0].set_ylim(-2.4, 2.4)
        self._save(fig, "prior_posterior.pdf")

    # ---------------------------------------------------------------- slide 5
    def evidence(self) -> None:
        x = self.random.uniform(0, 4, 60)
        y = np.sin(1.5 * x) + self.random.normal(0, 0.3, len(x))
        spline = Spline(0.0, 4.0, 16)
        model = BayesianSpline(spline, x, y)
        strengths = 10 ** np.linspace(-3, 6, 80)
        evidence = [model.fit(s)[2] for s in strengths]
        best = model.best(strengths)
        picks = [(1e-3, "λ too small"), (best, "best λ"), (1e6, "λ too large")]
        fig = plt.figure(figsize=(7.4, 3.9))
        main = fig.add_axes((0.1, 0.17, 0.5, 0.72))
        main.plot(strengths, evidence, color=INK, linewidth=2.2)
        main.set_xscale("log")
        main.set_xlabel("λ (strength of the smoothness belief)")
        main.set_ylabel("evidence")
        main.set_yticks([])
        grid_x = np.linspace(0, 4, 200)
        weights = spline.weights(grid_x)
        for row, (strength, label) in enumerate(picks):
            colour = RED if label == "best λ" else AMBER
            value = model.fit(strength)[2]
            main.plot(strength, value, "o", color=colour, markersize=8, zorder=3)
            offset, align = {"λ too small": ((6, 8), "left"), "best λ": ((0, 9), "center"),
                             "λ too large": ((0, 9), "right")}[label]
            main.annotate(label, (strength, value), xytext=offset, textcoords="offset points",
                          ha=align, fontsize=11, color=colour)
            inset = fig.add_axes((0.7, 0.68 - row * 0.29, 0.28, 0.24))
            inset.plot(x, y, "o", color=LGREY, markersize=2.5)
            inset.plot(grid_x, weights @ model.fit(strength)[0], color=colour, linewidth=2)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=10.5, color=colour, loc="left", pad=2)
        self._save(fig, "evidence.pdf")

    # ---------------------------------------------------------------- slide 6
    def surface(self) -> None:
        income = np.linspace(0, 4, 50)
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
        for savings, colour in zip([1, 4, 7], [BLUE, GREY, RED]):
            truth = 0.5 * income + 0.25 * savings + 0.15 * income * savings
            curves = 0.5 * income + 0.25 * savings + 0.15 * (2 * savings + 4 * income - 8)
            axes[0].plot(income, truth, color=colour, linewidth=2.4, label=f"savings = {savings}")
            axes[1].plot(income, curves, color=colour, linewidth=2.4)
        axes[0].set_title("truth: the slope depends on savings", fontsize=12, color=INK, loc="left")
        axes[1].set_title("sum of curves: only parallel lines", fontsize=12, color=INK, loc="left")
        axes[0].legend(fontsize=10.5, loc="upper left")
        for ax in axes[:2]:
            ax.set_xlabel("income")
            ax.set_ylim(-0.9, 8.2)
        axes[0].set_ylabel("Y's score")
        axes[1].set_yticklabels([])
        grid1, grid2 = np.meshgrid(np.linspace(0, 4, 80), np.linspace(0, 8, 80))
        surface = 0.15 * (grid1 - 2) * (grid2 - 4)
        mesh = axes[2].imshow(surface, cmap=DIVERGING, vmin=-1.2, vmax=1.2, origin="lower",
                              extent=(0, 4, 0, 8), aspect="auto", interpolation="bilinear")
        fig.colorbar(mesh, ax=axes[2], label="push on Y's score")
        axes[2].set_title("the surface: what curves miss", fontsize=12, color=INK, loc="left")
        axes[2].set_xlabel("income")
        axes[2].set_ylabel("savings")
        axes[2].spines["top"].set_visible(True)
        axes[2].spines["right"].set_visible(True)
        self._save(fig, "surface.pdf")

    # ---------------------------------------------------------------- slide 7
    def fast(self) -> None:
        income = self.random.uniform(0, 4, 700)
        savings = self.random.uniform(0, 8, 700)
        leftover = 0.15 * (income - 2) * (savings - 4) + self.random.normal(0, 0.25, 700)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(income, savings, c=leftover, cmap=DIVERGING, vmin=-0.9, vmax=0.9, s=14, linewidths=0)
        cut1, cut2 = np.median(income), np.median(savings)
        ax.axvline(cut1, color=INK, linewidth=1.6)
        ax.axhline(cut2, color=INK, linewidth=1.6)
        for low1, high1, x_text in ((0, cut1, cut1 / 2), (cut1, 4, (cut1 + 4) / 2)):
            for low2, high2, y_text in ((0, cut2, cut2 / 2), (cut2, 8, (cut2 + 8) / 2)):
                inside = (income >= low1) & (income < high1) & (savings >= low2) & (savings < high2)
                ax.text(x_text, y_text, f"{leftover[inside].mean():+.2f}", ha="center", va="center",
                        fontsize=19, fontweight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_xlabel("income")
        ax.set_ylabel("savings")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 8)
        self._save(fig, "fast.pdf")

    @staticmethod
    def _save(fig, name: str) -> None:
        fig.savefig(ASSETS / name, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


if __name__ == "__main__":
    Style.apply()
    Figures().all()
    print(f"figures saved in {ASSETS}")
