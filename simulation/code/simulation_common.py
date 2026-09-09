"""Shared numerical routines for the two manuscript simulations."""

from __future__ import annotations

import numpy as np


def k_rbf(x, y, ell):
    """RBF kernel evaluated between one-dimensional input arrays."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(1, -1)
    squared_distance = (x - y) ** 2
    return np.exp(-0.5 * squared_distance / (ell**2))


def zscore(values):
    """Population-standardize a one-dimensional array."""
    values = np.asarray(values)
    sd = np.std(values)
    if sd == 0:
        return values * 0.0
    return (values - np.mean(values)) / sd


def solve_spd(matrix, rhs):
    """Solve a positive-definite system, with a general solve as fallback."""
    try:
        cholesky = np.linalg.cholesky(matrix)
        intermediate = np.linalg.solve(cholesky, rhs)
        return np.linalg.solve(cholesky.T, intermediate)
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix, rhs)


def krr_fit_coeffs(x_train, y_train, x_basis, ell, regularization):
    """Solve exact dual KRR using the training inputs as kernel centres."""
    if not np.array_equal(np.asarray(x_train), np.asarray(x_basis)):
        raise ValueError("Exact dual KRR requires the training inputs as centres")
    if regularization <= 0:
        raise ValueError("The regularization coefficient must be positive")
    kernel = k_rbf(x_train, x_train, ell)
    system = kernel + len(x_train) * regularization * np.eye(len(x_train))
    return solve_spd(system, y_train)


def reconstruct_from_dual_pca(values, component_count):
    """Reconstruct rows of a data matrix from a truncated dual PCA."""
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    gram = centered @ centered.T
    eigenvalues, participant_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    participant_vectors = participant_vectors[:, order]
    retained = min(component_count, np.sum(eigenvalues > 1e-12))
    eigenvalues = eigenvalues[:retained]
    participant_vectors = participant_vectors[:, :retained]
    if retained == 0:
        return np.broadcast_to(mean, values.shape).copy()
    scores = participant_vectors * np.sqrt(eigenvalues)
    loadings = (
        np.diag(1.0 / np.sqrt(eigenvalues))
        @ participant_vectors.T
        @ centered
    )
    return scores @ loadings + mean


def metric_pca(coefficients, metric, component_count):
    """PCA of coefficient rows under a supplied positive metric matrix."""
    mean = coefficients.mean(axis=0, keepdims=True)
    centered = coefficients - mean
    gram = centered @ metric @ centered.T
    eigenvalues, participant_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    participant_vectors = participant_vectors[:, order]
    retained = min(component_count, np.sum(eigenvalues > 1e-12))
    eigenvalues = eigenvalues[:retained]
    participant_vectors = participant_vectors[:, :retained]
    if retained == 0:
        component_rows = np.zeros((0, coefficients.shape[1]))
        reconstructed = np.broadcast_to(mean, coefficients.shape).copy()
    else:
        scores = participant_vectors * np.sqrt(eigenvalues)
        component_rows = (
            np.diag(1.0 / np.sqrt(eigenvalues))
            @ participant_vectors.T
            @ centered
        )
        reconstructed = scores @ component_rows + mean
    return reconstructed, component_rows, eigenvalues


def make_block_folds(observation_count, fold_count):
    """Split ordered observations into contiguous, nearly equal blocks."""
    sizes = [
        observation_count // fold_count
        + (1 if index < observation_count % fold_count else 0)
        for index in range(fold_count)
    ]
    indices = np.arange(observation_count)
    folds = []
    start = 0
    for size in sizes:
        folds.append(indices[start : start + size])
        start += size
    return folds


def row_zscore(values):
    """Population-standardize each row of a two-dimensional array."""
    means = values.mean(axis=1, keepdims=True)
    sds = values.std(axis=1, keepdims=True)
    sds = np.where(sds == 0.0, 1.0, sds)
    return (values - means) / sds


def xcv_zrmse_score_session(
    x_obs,
    observations,
    ell,
    regularization,
    folds,
):
    """Mean negative RMSE after participant-wise training-fold standardization.

    Each fold uses its training mean and population SD to transform both
    training and held-out responses. KRR is solved in its exact dual form.
    """
    observation_count = len(x_obs)
    kernel = k_rbf(x_obs, x_obs, ell)
    fold_scores = []

    for test_indices in folds:
        train_indices = np.setdiff1d(np.arange(observation_count), test_indices)
        train = observations[:, train_indices]
        train_mean = train.mean(axis=1, keepdims=True)
        train_sd = train.std(axis=1, keepdims=True)
        if np.any(train_sd <= 0):
            raise ValueError("Training-fold response SD must be positive")
        standardized_train = (train - train_mean) / train_sd
        standardized_test = (observations[:, test_indices] - train_mean) / train_sd
        system = kernel[np.ix_(train_indices, train_indices)] + (
            len(train_indices) * regularization * np.eye(len(train_indices))
        )
        coefficients = solve_spd(system, standardized_train.T)
        prediction = (kernel[np.ix_(test_indices, train_indices)] @ coefficients).T
        errors = np.sqrt(
            np.mean(
                (standardized_test - prediction) ** 2,
                axis=1,
            )
        )
        fold_scores.extend((-errors).tolist())

    return float(np.mean(fold_scores))
