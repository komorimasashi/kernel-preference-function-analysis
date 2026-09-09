"""Check dual KRR against a feature-space ridge solution and PCA invariants.

Run with ``python empirical/code/validate_model.py`` from the release root.
The independent feature-space construction also tests unequal observation
counts, multivariate inputs, and repeated stimulus positions.
"""

import unittest

import numpy as np

from model.rkhs_pca import RKHSFunctionPCA, dual_krr_coefficients, rbf_kernel


class ModelChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(47012)
        self.x = [rng.uniform(-1, 1, (n, 2)) for n in (4, 6, 5, 7)]
        self.x[1][1] = self.x[1][0]
        self.y = [rng.normal(size=len(x)) for x in self.x]
        self.params = {"length": 0.43, "beta": 0.027}
        self.model = RKHSFunctionPCA(self.x, self.y, self.params, modelDim=3).fit()

        # Each row of the Cholesky factor is an explicit feature vector with
        # the required kernel inner products. Solve ridge regression in that
        # feature space, independently of the participant-specific dual solve.
        self.centers = np.unique(np.concatenate(self.x), axis=0)
        kernel = rbf_kernel(self.centers, self.centers, self.params["length"])
        self.cholesky = np.linalg.cholesky(kernel)
        self.weights = []
        for x, y in zip(self.x, self.y):
            features = self.features(x)
            weight = np.linalg.solve(
                features.T @ features + len(x) * self.params["beta"] * np.eye(len(self.centers)),
                features.T @ y,
            )
            self.weights.append(weight)
        self.weights = np.asarray(self.weights)
        self.query = rng.uniform(-1, 1, (13, 2))

    def features(self, x):
        return np.linalg.solve(
            self.cholesky, rbf_kernel(self.centers, x, self.params["length"])
        ).T

    def test_dual_predictions_and_rkhs_geometry_match_feature_ridge(self):
        expected = self.weights @ self.features(self.query).T
        np.testing.assert_allclose(self.model.predict(self.query), expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(self.model.H, self.weights @ self.weights.T, rtol=0, atol=1e-10)

    def test_pca_matches_feature_space_svd_and_preserves_full_rank_functions(self):
        centered = self.weights - self.weights.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        np.testing.assert_allclose(self.model.eVal[:3], singular_values[:3]**2, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            self.model.predict_approx(self.query), self.model.predict(self.query), rtol=0, atol=1e-10
        )
        basis = self.model.eVec[:, :3].T @ centered / np.sqrt(self.model.eVal[:3])[:, None]
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            self.model.pc_functions(self.query), basis @ self.features(self.query).T, rtol=0, atol=1e-10
        )
        score_distance = np.sum((self.model.Z[:, None] - self.model.Z[None, :])**2, axis=2)
        feature_distance = np.sum((self.weights[:, None] - self.weights[None, :])**2, axis=2)
        np.testing.assert_allclose(score_distance, feature_distance, rtol=0, atol=1e-10)

    def test_truncated_reconstruction_error_equals_discarded_variance(self):
        model = RKHSFunctionPCA(self.x, self.y, self.params, modelDim=2).fit()
        centered = self.weights - self.weights.mean(axis=0)
        projected = model.eVec[:, :2] @ model.eVec[:, :2].T @ centered
        self.assertAlmostEqual(float(np.sum((centered - projected)**2)), float(model.eVal[2:].sum()), places=10)

    def test_positive_rank_and_regularization_requirements(self):
        with self.assertRaises(ValueError):
            RKHSFunctionPCA([self.x[0]]*3, [self.y[0]]*3, self.params, modelDim=1).fit()
        with self.assertRaises(ValueError):
            dual_krr_coefficients(self.x[0], self.y[0], self.params["length"], beta=0)
        with self.assertRaises(ValueError):
            RKHSFunctionPCA(self.x, self.y, self.params, modelDim=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
