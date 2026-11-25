from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CDNODSettings:
    """The arguments of nonsta_cd_new.m and its `pars`, with the values of the
    authors' examples.

    width: kernel width for the observed variables; 0 lets each step pick its own
        default for the sample size (pars.width).
    width_time: kernel width for the time index in the conditional tests; 0 means
        0.1 there too (pars.widthT).
    gp_tests: learn the kernel widths of the conditional tests with a Gaussian
        process (pars.if_GP1).
    gp_directions: the same in the Independent Change Principle step (pars.if_GP2).
    random_features: indtest_new_t_RFF instead of indtest_new_t.
    seed: for the χ² draws and the random features (MATLAB's own random numbers
        cannot be reproduced, so p-values differ from MATLAB's by simulation noise).
    """

    alpha: float = 0.05
    max_fan_in: int = 2
    width: float = 0.0
    width_time: float = 0.1
    gp_tests: bool = True
    gp_directions: bool = True
    random_features: bool = False
    seed: int = 0

    @classmethod
    def for_rows(cls, rows: int, **changes) -> "CDNODSettings":
        """The rule in the authors' examples: up to 1,000 rows, KCI with widths
        learnt by a Gaussian process; up to 2,000, KCI with fixed widths; beyond,
        the random-feature KCI with fixed widths."""
        if rows <= 1000:
            base = cls(gp_tests=True, random_features=False)
        elif rows <= 2000:
            base = cls(gp_tests=False, random_features=False)
        else:
            base = cls(gp_tests=False, random_features=True)
        return replace(base, **changes)
