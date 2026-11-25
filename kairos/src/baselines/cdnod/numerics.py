"""The MATLAB building blocks the CD-NOD code relies on, with MATLAB's conventions:
columns standardised with the n - 1 spread, eigenvalues largest first, inverses
through a Cholesky factor. Each method names the MATLAB file it ports.

Linear algebra goes through numpy only: in the installed scipy (1.18.1, Accelerate)
linalg.cholesky(lower=True) leaves junk above the diagonal and crashes the process
on 2,000 × 2,000 matrices."""

import numpy as np


class Numerics:

    @staticmethod
    def standardise(x: np.ndarray) -> np.ndarray:
        """Each column to mean 0 and spread 1; the spread divides by n - 1, as MATLAB's std."""
        return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

    @staticmethod
    def squared_distances(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        """dist2.m: squared distances between the rows of a and the rows of b."""
        b = a if b is None else b
        distances = (a ** 2).sum(axis=1)[:, None] + (b ** 2).sum(axis=1)[None, :] - 2 * a @ b.T
        return np.maximum(distances, 0)

    @staticmethod
    def gaussian_kernel(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
        """kernel.m with [theta, 1]: exp(-theta/2 * squared distance)."""
        return np.exp(-Numerics.squared_distances(a, b) * theta / 2)

    @staticmethod
    def se_kernel(x: np.ndarray, lengths: np.ndarray, signal: float) -> np.ndarray:
        """covSEard.m: a Gaussian kernel with one length scale per column, times signal²."""
        return signal ** 2 * np.exp(-Numerics.squared_distances(x / lengths) / 2)

    @staticmethod
    def top_eigen(matrix: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        """eigdec.m: the `count` largest eigenvalues of a symmetric matrix, largest
        first, with their eigenvectors. (eigdec switches to eigs, largest magnitude,
        when count is under 4% of the size; for the positive semidefinite kernel
        matrices it gets, that is the same set.)"""
        values, vectors = np.linalg.eigh(matrix)
        return values[::-1][:count], vectors[:, ::-1][:, :count]

    @staticmethod
    def pd_inverse(matrix: np.ndarray) -> np.ndarray:
        """pdinv.m: inverse through the Cholesky factor, or through the SVD when the
        matrix is not positive definite. With A = LL' (L = U' in MATLAB's chol),
        A⁻¹ = L⁻ᵀL⁻¹."""
        try:
            lower = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            u, s, vt = np.linalg.svd(matrix)
            return vt.T @ np.diag(1 / s) @ u.T
        inverse_lower = np.linalg.inv(lower)
        return inverse_lower.T @ inverse_lower

    @staticmethod
    def centring(rows: int) -> np.ndarray:
        """H = I - 11'/T, which centres a kernel matrix in feature space."""
        return np.eye(rows) - np.ones((rows, rows)) / rows

    @staticmethod
    def is_time_index(column: np.ndarray) -> bool:
        """The check in CInd_test_new_withGP_t(_RFF).m: is this standardised column
        the time index 1, ..., T?"""
        time = Numerics.standardise(np.arange(1, len(column) + 1, dtype=float)[:, None])[:, 0]
        return bool(np.linalg.norm(column - time) < 1e-5)

    @staticmethod
    def product_spectrum(values_x: np.ndarray, vectors_x: np.ndarray, values_y: np.ndarray,
                         vectors_y: np.ndarray, threshold: float) -> np.ndarray:
        """The weights of the conditional KCI null distribution (CInd_test_new_withGP_t.m):
        the eigenvalues of uu'uu, where each column of uu is the elementwise product of
        an eigenvector of Kx|z and one of Ky|z, each scaled by the root of its eigenvalue.
        When uu has more columns than rows, uu·uu' = (AxAx') ∘ (AyAy') is formed
        directly: the same matrix without building uu."""
        keep_x = values_x > values_x.max() * threshold
        keep_y = values_y > values_y.max() * threshold
        scaled_x = vectors_x[:, keep_x] * np.sqrt(values_x[keep_x])
        scaled_y = vectors_y[:, keep_y] * np.sqrt(values_y[keep_y])
        rows, size = len(scaled_x), scaled_x.shape[1] * scaled_y.shape[1]
        if size > rows:
            product = (scaled_x @ scaled_x.T) * (scaled_y @ scaled_y.T)
        else:
            uu = (scaled_x[:, :, None] * scaled_y[:, None, :]).reshape(rows, size)
            product = uu.T @ uu
        values = np.linalg.eigvalsh(product)[::-1]
        return values[values > values.max() * threshold]
