"""indtest_new_t_RFF.m: KCI with random Fourier features, the authors' choice above
2,000 rows. Widths are fixed (no Gaussian process).

Where MATLAB takes eigenvalues of a T × T product of the features, Z'Z, this port
takes them from the features directly (from ZZ', or from the SVD of Z): the same
non-zero eigenvalues and eigenvectors, without forming the T × T matrix.

KCI-test code: Copyright (c) 2010-2011 Kun Zhang, Jonas Peters; BSD-style licence
(KCI-test/COPYING): redistribution allowed, keeping this notice."""

import numpy as np

from src.baselines.cdnod.null_distribution import ChiSquareMixture
from src.baselines.cdnod.numerics import Numerics
from src.baselines.cdnod.random_features import RandomFeatures
from src.baselines.cdnod.settings import CDNODSettings


class RandomFeatureCITest:

    def __init__(self, settings: CDNODSettings, random: np.random.Generator):
        self.settings = settings
        self.null = ChiSquareMixture(random)
        self.features = RandomFeatures(random)

    def p_value(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        if z.shape[1] == 0:
            return self.unconditional(x, y)
        return self.conditional(x, y, z)

    def unconditional(self, x: np.ndarray, y: np.ndarray) -> float:
        """UInd_KCItest_RFF.m: the squared norm of the features' cross-covariance."""
        rows = len(y)
        eigen_count = rows // 2 if rows > 1000 else rows
        x, y = Numerics.standardise(x), Numerics.standardise(y)
        width = self.settings.width or (0.8 if rows < 200 else 0.5 if rows < 1200 else 0.3)
        zx = self.features.transform(x / width)
        zy = self.features.transform(y / width)
        cross = (zx - zx.mean(axis=1, keepdims=True)) @ (zy - zy.mean(axis=1, keepdims=True)).T
        statistic = float(np.sum(cross ** 2))
        values_x = self._gram_eigenvalues(zx, eigen_count)  # eigdec(Zx'*Zx, Num_eig)
        values_y = self._gram_eigenvalues(zy, eigen_count)
        products = np.outer(values_y, values_x).ravel()
        products = products[products > products.max() * 1e-6]
        return self.null.p_value(products / rows, statistic, rows, draws=5000, keep_remainder=True)

    def conditional(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """CInd_test_new_withGP_t_RFF.m: features of (x, z/2) and of y, minus what a
        ridge regression on z's features explains."""
        rows, columns = z.shape
        x, y, z = Numerics.standardise(x), Numerics.standardise(y), Numerics.standardise(z)
        width = self.settings.width or (1.2 if rows <= 200 else 0.6 if rows < 1200 else 0.4)
        width_time = self.settings.width_time or 0.1
        scale = width * np.sqrt(columns) / np.sqrt(2)
        zx = self._centred(self.features.transform(np.hstack([x, z / 2]) / scale))
        zy = self._centred(self.features.transform(y / scale))
        if Numerics.is_time_index(z[:, -1]):
            time = z[:, -1:] / (width_time / np.sqrt(2))
            if columns > 1:
                time = np.hstack([z[:, :-1] / (width * np.sqrt(columns - 1) / np.sqrt(2)), time])
            zz = self.features.transform(time)
        else:
            zz = self.features.transform(z / scale)
        zz = self._centred(zz)
        ridge = zz @ zz.T + 1e-5 * np.eye(len(zz))
        ex = self._centred(zx - np.linalg.solve(ridge, zz @ zx.T).T @ zz)
        ey = self._centred(zy - np.linalg.solve(ridge, zz @ zy.T).T @ zz)
        statistic = float(np.sum((ex @ ey.T) ** 2))
        values_x, vectors_x = self._gram_eigen(ex)  # eigdec(Ex'*Ex, T)
        values_y, vectors_y = self._gram_eigen(ey)
        weights = Numerics.product_spectrum(values_x, vectors_x, values_y, vectors_y, 1e-5)
        return self.null.p_value(weights, statistic, rows, draws=10000, keep_remainder=False)

    @staticmethod
    def _centred(features: np.ndarray) -> np.ndarray:
        return features - features.mean(axis=1, keepdims=True)

    @staticmethod
    def _gram_eigenvalues(features: np.ndarray, count: int) -> np.ndarray:
        """The `count` largest eigenvalues of Z'Z, from the smaller ZZ'; the rest are zero."""
        values = np.linalg.eigvalsh(features @ features.T)[::-1]
        return np.maximum(values[:count], 0)

    @staticmethod
    def _gram_eigen(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of Z'Z from the SVD Z = USV': S² and V."""
        _, singular, vt = np.linalg.svd(features, full_matrices=False)
        return singular ** 2, vt.T
