"""The smallest noise variance an equation is trusted to have."""

import numpy as np


class NoiseFloor:
    """The noise floor of one equation, in the child's units squared.

    Every continuous variable is trusted only up to an imprecision of `share` × its
    own variance (with 0.0001, a variable of spread 10 is trusted up to ±0.1). An equation
    inherits the imprecision of everything in it: the child's own, plus each parent's
    carried through the equation by how steep it is there,

        floor = share × var(child) + Σ_parent slope² × share × var(parent)

    This is the usual propagation of measurement error. A steep curve turns a small
    wobble of the parent into a large wobble of the child.

    Why it is carried, and not just the child's own share: an exact formula can be
    solved for any of its variables (z = x − y, or x = z + y), and it says the same
    thing each time. With the child's share alone the floor would be 1% of var(z) or
    1% of var(x) (say the share is 1%), so the rewriting whose child varies least would
    win, whatever causes what. Carried, it is 1% of var(x) + var(y) + var(z) each time.

    How large the share may be: the floor is added to the measured noise, so it must be
    small next to the noise of an ordinary relation, or it blurs what tells the
    directions apart. On synthetic pairs the learnt noise is unharmed up to 0.0003 and
    loses most directions at 0.01.

    A categorical parent is taken as recorded exactly: it has no slope.

    `parents` has one entry per continuous parent: its imprecision, and the matrix
    that turns the equation's coefficients into its slope at every row. Without
    entries the floor is the child's own share.
    """

    def __init__(self, share: float, parents: list[tuple[float, np.ndarray]] | None = None):
        self.share = share
        self.parents = parents if parents is not None else []

    def rows(self, coefficients: np.ndarray, target: np.ndarray) -> np.ndarray:
        """The floor at each row (a curve is steeper at some rows than at others)."""
        floor = np.full(len(target), self.share * np.var(target))
        for imprecision, slopes in self.parents:
            floor += imprecision * (slopes @ coefficients) ** 2
        return floor
