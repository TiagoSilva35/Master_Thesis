"""A continuous node whose noise spread and shape are learnt."""

import numpy as np

from src.causal_model.nodes.fitting import SmoothnessBlock
from src.causal_model.nodes.node import Node
from src.causal_model.nodes.noise_density import NoiseDensity
from src.causal_model.nodes.noise_spread import NoiseSpread


class FlexibleNode(Node):
    """A continuous node whose noise is learnt instead of assumed: a spread that
    can change with the parents, and a shape that need not be a bell.

        x = constant + Σ f_parent(x_parent) + spread(parents) × noise

    1. The mean curves are fitted exactly as a Node's, with λ chosen by the
       bell-shaped evidence. Leftovers r = x − prediction (a root's: x − its average).
    2. NoiseSpread fits the spread curves to r² (log v = c + Σ g_parent, pulled
       towards flat). The noise floor is added to each v, carried through the
       curves at that row (see NoiseFloor). A root's spread is one constant.
    3. z = r / √v, the leftovers with each person's spread taken out, should look
       alike for everyone; NoiseDensity learns their shape f (pulled towards a bell).
    4. evidence = Σ [log f(z_i) − ½ log v_i] − the mean curves' prior cost
                  + their Occam term + the spread curves' Occam term + f's Occam term

       The −½ log v is there because dividing by √v squeezes the axis.

    Both new parts are nested: a flat spread and a bell shape give back the plain
    node's model, and the evidence decides per variable how far to move. Under a
    location-scale noise model the true direction is where z is unrelated to the
    causes; scoring the wrong direction wastes probability on pairs that never
    happen, which f can see (Kpotufe et al. 2014; Immer et al. 2023).

    Bins for z are never narrower than two scales, so an exact or rounded formula
    cannot earn spikes:
    - the floor's: share × √(2πe), where share = c / (c + mean v) is how much of
      the leftovers' variance the floor (c on average) adds. While the floor is negligible the
      bins stay fine; when it dominates, the narrowest shape allowed is a box as
      uncertain as a bell of spread 1 (√(2πe) wide), which gives back the bell
      floor's cap: no arrow gains more than about N/2 · log(1 / noise_floor).
    - the child's rounding step, divided by the typical spread √(mean(v + c)).

    The spread is fitted once, after the mean curves; refitting the mean with
    weights 1/v until both settle (the joint fit) is left for later.
    """

    density: NoiseDensity | None = None  # f, set when the node is fitted
    spread: NoiseSpread | None = None    # the spread curves, set when the node is fitted

    def refit(self, data: dict[str, np.ndarray]) -> None:
        super().refit(data)
        values = data[self.name]
        leftovers = values - self.predict(data)
        self.spread = NoiseSpread(self.fitting)
        self.spread.fit(self.design(data), self.spread_blocks(), leftovers)
        variance = self.spread.variance + self.row_floor
        standardised = leftovers / np.sqrt(variance)
        self.density = NoiseDensity(smallest_width=self._smallest_width(values, variance))
        self.density.fit(standardised)
        self.evidence = float(np.sum(self.density.log_density(standardised) - 0.5 * np.log(variance))
                              - self.prior_cost + self.occam + self.spread.occam + self.density.occam)

    def _smallest_width(self, values: np.ndarray, variance: np.ndarray) -> float:
        share = np.mean(self.row_floor) / np.mean(variance)  # the floor's part of the leftovers' variance
        return max(share * np.sqrt(2 * np.pi * np.e), self.rounding_step(values) / np.sqrt(np.mean(variance)))

    def spread_blocks(self) -> list[SmoothnessBlock]:
        """One flattening block per curve (and per surface): a strong pull makes the
        spread flat."""
        columns = self.columns()
        blocks = [SmoothnessBlock(f"spread {parent}", columns[parent], self.curves[parent].flattening())
                  for parent in self.parents]
        for pair in self.pairs:
            label = self.surface_label(pair)
            blocks.append(SmoothnessBlock(f"spread {label}", columns[label], self.surfaces[pair].flattening()))
        return blocks

    @staticmethod
    def rounding_step(values: np.ndarray) -> float:
        """The step of the grid the values lie on (0.01 for values rounded to
        two decimals), or 0 when they do not lie on one."""
        unique = np.unique(values)
        if len(unique) < 2:
            return 0.0
        gaps = np.diff(unique)
        step = float(gaps.min())
        multiples = gaps / step
        return step if np.all(np.abs(multiples - np.round(multiples)) < 0.01) else 0.0
