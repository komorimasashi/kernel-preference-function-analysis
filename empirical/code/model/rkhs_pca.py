"""Participant-specific dual KRR and function PCA in the kernel's RKHS.

The implementation follows the manuscript's Method section. No common-basis
inverse, coefficient penalty, or additional diagonal jitter is used.
"""

import numpy as np


def input_matrix(values):
    """Represent scalar or multivariate observations as an N-by-d matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Inputs must be a finite N-by-d matrix.")
    return values


def rbf_kernel(x1, x2, length):
    x1, x2 = input_matrix(x1), input_matrix(x2)
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input dimensions must agree.")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("The kernel length scale must be positive and finite.")
    squared_distance = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    return np.exp(-squared_distance / (2 * float(length) ** 2))


def dual_krr_coefficients(x, y, length, beta):
    """Solve (K_tt + N_t beta I) alpha_t = s_t without extra jitter."""
    x = input_matrix(x)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(x) != len(y) or not np.all(np.isfinite(y)):
        raise ValueError("Each participant needs matching, finite observations.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("The regularization coefficient beta must be positive.")
    kernel = rbf_kernel(x, x, length)
    return np.linalg.solve(kernel + len(x) * beta * np.eye(len(x)), y)


class RKHSFunctionPCA:
    """Fit functions on their own observations, then perform RKHS-PCA.

    ``params`` contains the shared ``length`` and ``beta``; beta multiplies
    the squared RKHS norm in the mean-squared-loss objective. Observation
    counts and positions may differ across participants. The score matrix Z
    has participants in rows (the transpose of the manuscript's Z).
    """

    def __init__(self, x_list, y_list, params, modelDim=1):
        self.x_list = [input_matrix(x) for x in x_list]
        self.y_list = [np.asarray(y, dtype=np.float64).reshape(-1) for y in y_list]
        self.task_size = len(self.x_list)
        if self.task_size < 2 or self.task_size != len(self.y_list):
            raise ValueError("Provide observations for at least two participants.")
        self.params = {"length": float(params["length"]), "beta": float(params["beta"])}
        self.L = int(modelDim)
        if self.L != modelDim or not 1 <= self.L < self.task_size:
            raise ValueError("Retain between 1 and T-1 principal components.")
        self.alphas = [
            dual_krr_coefficients(x, y, **self.params)
            for x, y in zip(self.x_list, self.y_list)
        ]
        self.H = np.empty((self.task_size, self.task_size), dtype=np.float64)
        for t in range(self.task_size):
            for other in range(t, self.task_size):
                cross_kernel = rbf_kernel(
                    self.x_list[t], self.x_list[other], self.params["length"]
                )
                value = self.alphas[t] @ cross_kernel @ self.alphas[other]
                self.H[t, other] = self.H[other, t] = value
        self.eVal = None
        self.eVec = None
        self.Z = None

    def predict_subject(self, x_new, s_idx):
        t = int(s_idx)
        return rbf_kernel(x_new, self.x_list[t], self.params["length"]) @ self.alphas[t]

    def predict(self, x_new):
        return np.vstack([self.predict_subject(x_new, t) for t in range(self.task_size)])

    def fit(self):
        centering = np.eye(self.task_size) - np.ones_like(self.H) / self.task_size
        centered_gram = centering @ self.H @ centering
        centered_gram = (centered_gram + centered_gram.T) / 2
        values, vectors = np.linalg.eigh(centered_gram)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # Numerical rank excludes zero eigenvalues and roundoff at zero.
        scale = max(np.max(np.abs(self.H)), np.max(np.abs(values)))
        tolerance = self.task_size * np.finfo(float).eps * scale
        if values[-1] < -tolerance:
            raise ArithmeticError("The function Gram matrix is not positive semidefinite.")
        values[values <= tolerance] = 0.0
        if self.L > np.count_nonzero(values):
            raise ValueError("Requested components exceed the positive numerical rank.")
        self.eVal, self.eVec = values, vectors
        self.Z = vectors[:, :self.L] * np.sqrt(values[:self.L])
        return self

    def pc_functions(self, x_new):
        """Evaluate RKHS-orthonormal principal functions; return L-by-N_new."""
        if self.Z is None:
            raise RuntimeError("Call fit() before evaluating principal components.")
        curves = self.predict(x_new)
        centered_curves = curves - curves.mean(axis=0, keepdims=True)
        return (self.eVec[:, :self.L].T @ centered_curves) / np.sqrt(self.eVal[:self.L])[:, None]

    def generate_pos(self, x_new, Z):
        scores = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        if scores.shape[1] != self.L:
            raise ValueError("Each score vector must contain L components.")
        return self.predict(x_new).mean(axis=0) + scores @ self.pc_functions(x_new)

    def predict_approx(self, x_new):
        if self.Z is None:
            raise RuntimeError("Call fit() before reconstructing functions.")
        return self.generate_pos(x_new, self.Z)
