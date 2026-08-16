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


def krr_fit_coeffs(x_train, y_train, x_basis, ell, regularization, jitter=1e-6):
    """Fit KRR coefficients on a fixed RBF-kernel basis."""
    k_train_basis = k_rbf(x_train, x_basis, ell)
    k_basis = k_rbf(x_basis, x_basis, ell)
    if jitter > 0:
        k_basis = k_basis + jitter * np.eye(k_basis.shape[0])
    system = (
        k_train_basis.T @ k_train_basis
        + (len(x_train) * regularization) * k_basis
    )
    return solve_spd(system, k_train_basis.T @ y_train)


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
    jitter=1e-6,
):
    """Mean negative z-RMSE for shared KRR parameters and x-block folds."""
    observation_count = len(x_obs)
    k_basis = k_rbf(x_obs, x_obs, ell) + jitter * np.eye(observation_count)
    fold_scores = []

    for test_indices in folds:
        train_indices = np.setdiff1d(np.arange(observation_count), test_indices)
        y_train = observations[:, train_indices]
        means = y_train.mean(axis=1, keepdims=True)
        sds = y_train.std(axis=1, keepdims=True)
        sds = np.where(sds == 0.0, 1.0, sds)
        standardized_train = (y_train - means) / sds

        k_train_basis = k_rbf(x_obs[train_indices], x_obs, ell)
        system = (
            k_train_basis.T @ k_train_basis
            + (len(train_indices) * regularization) * k_basis
        )
        coefficients = solve_spd(
            system, k_train_basis.T @ standardized_train.T
        ).T
        standardized_prediction = (
            k_rbf(x_obs[test_indices], x_obs, ell) @ coefficients.T
        ).T
        prediction = means + sds * standardized_prediction
        errors = np.sqrt(
            np.mean(
                (
                    row_zscore(observations[:, test_indices])
                    - row_zscore(prediction)
                )
                ** 2,
                axis=1,
            )
        )
        fold_scores.extend((-errors).tolist())

    return float(np.mean(fold_scores))
