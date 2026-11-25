"""indtest_new_t.m: the kernel-based conditional independence (KCI) test of
Zhang, Peters, Janzing and Schölkopf (UAI 2011), as CD-NOD calls it.

KCI-test code: Copyright (c) 2010-2011 Kun Zhang, Jonas Peters; BSD-style licence
(KCI-test/COPYING): redistribution allowed, keeping this notice."""

from typing import Protocol

import numpy as np

from src.baselines.cdnod.gaussian_process import GaussianProcessWidths
from src.baselines.cdnod.null_distribution import ChiSquareMixture
from src.baselines.cdnod.numerics import Numerics
from src.baselines.cdnod.settings import CDNODSettings


class IndependenceTest(Protocol):

    def p_value(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """p-value of 'x is independent of y given z'; z may have no columns."""
        ...


class KernelCITest:
    """indtest_new_t.m: the unconditional test when z is empty, else the conditional
    one. (The wrapper's `pairwise` option, off in every example, is not ported.)"""

    def __init__(self, settings: CDNODSettings, random: np.random.Generator,
                 widths: GaussianProcessWidths):
        self.settings = settings
        self.null = ChiSquareMixture(random)
        self.widths = widths

    def p_value(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        if z.shape[1] == 0:
            return self.unconditional(x, y)
        return self.conditional(x, y, z)

    def unconditional(self, x: np.ndarray, y: np.ndarray) -> float:
        """UInd_KCItest.m: the HSIC statistic tr(KxKy) of the centred kernel matrices."""
        rows = len(y)
        eigen_count = rows // 2 if rows > 1000 else rows
        kx, ky = self.unconditional_kernels(x, y)
        statistic = float(np.sum(kx * ky.T))  # tr(KxKy)
        values_x, _ = Numerics.top_eigen((kx + kx.T) / 2, eigen_count)
        values_y, _ = Numerics.top_eigen((ky + ky.T) / 2, eigen_count)
        products = np.outer(values_y, values_x).ravel()  # MATLAB's stack(): column by column
        products = products[products > products.max() * 1e-6]
        return self.null.p_value(products / rows, statistic, rows, draws=2000, keep_remainder=True)

    def unconditional_kernels(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The centred Gaussian kernel matrices of UInd_KCItest.m, with its width
        rule: 0.8 below 200 rows, 0.5 below 1,200, 0.3 from there."""
        rows = len(y)
        x, y = Numerics.standardise(x), Numerics.standardise(y)
        width = self.settings.width or (0.8 if rows < 200 else 0.5 if rows < 1200 else 0.3)
        theta = 1 / width ** 2
        h = Numerics.centring(rows)
        kx = h @ Numerics.gaussian_kernel(x, x, theta * x.shape[1]) @ h
        ky = h @ Numerics.gaussian_kernel(y, y, theta * y.shape[1]) @ h
        return kx, ky

    def conditional(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """CInd_test_new_withGP_t.m: tr(Kx|z Ky|z), where K·|z keeps what a kernel
        regression on z leaves unexplained."""
        rows, columns = z.shape
        x, y, z = Numerics.standardise(x), Numerics.standardise(y), Numerics.standardise(z)
        width = self.settings.width or (1.2 if rows <= 200 else 0.6 if rows < 1200 else 0.4)
        theta = 1 / (width ** 2 * columns)
        h = Numerics.centring(rows)
        kx = h @ Numerics.gaussian_kernel(np.hstack([x, z / 2]), np.hstack([x, z / 2]), theta) @ h
        ky = h @ Numerics.gaussian_kernel(y, y, theta) @ h
        if self.settings.gp_tests:
            residual_x = self._learnt_residual_maker(kx, z, width, min(400, rows // 4))
            residual_y = self._learnt_residual_maker(ky, z, width, min(200, rows // 5))
        else:
            residual_x = residual_y = self._fixed_residual_maker(z, width, theta, h)
        kxz = residual_x @ kx @ residual_x.T
        kyz = residual_y @ ky @ residual_y.T
        statistic = float(np.sum(kxz * kyz.T))  # tr(Kx|z Ky|z)
        values_x, vectors_x = Numerics.top_eigen((kxz + kxz.T) / 2, rows)
        values_y, vectors_y = Numerics.top_eigen((kyz + kyz.T) / 2, rows)
        weights = Numerics.product_spectrum(values_x, vectors_x, values_y, vectors_y, 1e-5)
        return self.null.p_value(weights, statistic, rows, draws=5000, keep_remainder=False)

    def _fixed_residual_maker(self, z: np.ndarray, width: float, theta: float, h: np.ndarray) -> np.ndarray:
        """I - Kz(Kz + λI)⁻¹ with λ = 0.001 and fixed widths; when z's last column is
        the time index, it gets its own width."""
        rows, columns = z.shape
        if Numerics.is_time_index(z[:, -1]):
            width_time = self.settings.width_time or 0.1
            lengths = np.array([width * np.sqrt(columns - 1)] * (columns - 1) + [width_time])
            kz = Numerics.se_kernel(z, lengths, 1.0)
        else:
            kz = Numerics.gaussian_kernel(z, z, theta)
        kz = h @ kz @ h
        return np.eye(rows) - kz @ Numerics.pd_inverse(kz + 1e-3 * np.eye(rows))

    def _learnt_residual_maker(self, k: np.ndarray, z: np.ndarray, width: float, eigen_count: int) -> np.ndarray:
        """I - Kz(Kz + σ²I)⁻¹ with Kz's widths and the noise σ learnt by regressing the
        leading eigenvectors of K on z with a Gaussian process."""
        rows, columns = z.shape
        values, vectors = Numerics.top_eigen((k + k.T) / 2, eigen_count)
        keep = values > values.max() * 1e-5
        values, vectors = values[keep], vectors[:, keep]
        targets = 2 * np.sqrt(rows) * vectors * np.sqrt(values) / np.sqrt(values[0])
        start = np.array([np.log(width * np.sqrt(columns))] * columns + [0.0, np.log(np.sqrt(0.1))])
        log_hyper = self.widths.fit(start, z, targets)
        kz = Numerics.se_kernel(z, np.exp(log_hyper[:columns]), float(np.exp(log_hyper[columns])))
        return np.eye(rows) - kz @ Numerics.pd_inverse(kz + np.exp(2 * log_hyper[-1]) * np.eye(rows))
