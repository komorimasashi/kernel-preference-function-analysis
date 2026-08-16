# model/KPCA.py
import numpy as np

class SparseKPCA:
    """
    各被験者 i の (X_i, y_i) から共通基底点 x_all 上のカーネル重み a_i を推定。
    (K + β Ki^T Ki)^{-1} (β Ki^T y) を Woodbury + Cholesky で高速に解く。
      K = kernel(x_all, x_all),  β = 1 / noise_level
    """

    def __init__(self, x_list, y_list, params, modelDim=1, basis_size=None, jitter=1e-6):
        self.x_list = [np.asarray(x, dtype=np.float64) for x in x_list]
        self.y_list = [np.asarray(y, dtype=np.float64).reshape(-1) for y in y_list]
        self.task_size = len(self.x_list)
        self.params = params
        self.L = int(modelDim)
        self.jitter = float(jitter)

        # 基底点（誘導点を使うなら basis_size を指定）
        Xcat = np.concatenate(self.x_list, axis=0)
        if basis_size is None or basis_size >= Xcat.shape[0]:
            self.x_all = Xcat
        else:
            rng = np.random.default_rng(0)
            idx = rng.choice(Xcat.shape[0], size=int(basis_size), replace=False)
            self.x_all = Xcat[idx]

        self.N, self.D = self.x_all.shape

        # 共通カーネル K と Cholesky
        self.K = self._rbf(self.x_all, self.x_all, self.params["length"])
        if self.jitter > 0.0:
            self.K += self.jitter * np.eye(self.N)
        self.K_chol = np.linalg.cholesky(self.K)  # K = L L^T

        # 各被験者の重み a
        self.a = self._calc_weight()

        # KPCA 用（fit() 後に利用）
        self.eVal = None
        self.eVec = None
        self.Z = None
        self.W = None
        self.W0 = None
        self.a_pos = None

    # ---- 基本関数 ----
    @staticmethod
    def _rbf(x1, x2, length):
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        x1n = np.sum(x1 * x1, axis=1)[:, None]
        x2n = np.sum(x2 * x2, axis=1)[None, :]
        dist2 = np.maximum(x1n + x2n - 2.0 * (x1 @ x2.T), 0.0)
        return np.exp(-dist2 / (2.0 * (float(length) ** 2)))

    def _solve_K(self, B):
        y = np.linalg.solve(self.K_chol, B)
        return np.linalg.solve(self.K_chol.T, y)

    def _calc_weight(self):
        beta = 1.0 / float(self.params["noise_level"])
        a = np.zeros((self.task_size, self.N), dtype=np.float64)
        for i in range(self.task_size):
            Xi, yi = self.x_list[i], self.y_list[i]
            if Xi.shape[0] == 0:
                continue
            Ki = self._rbf(Xi, self.x_all, self.params["length"])  # (n_i, N)
            Z  = self._solve_K(Ki.T)                                # (N, n_i) = K^{-1} Ki^T
            S  = Ki @ Z                                             # (n_i, n_i)
            # M = S + (1/β) I  を解く（SPD）
            M  = S + (1.0 / beta) * np.eye(S.shape[0])
            v1 = Z @ yi                  # (N,)
            v2 = Ki @ v1                 # (n_i,)
            v3 = np.linalg.solve(M, v2)  # (n_i,)
            a[i] = beta * (v1 - Z @ v3)  # (N,)
            #print(a.shape)
            #print(i+1)
        return a

    # ---- 予測 ----
    def predict_subject(self, x_new, s_idx):
        Kn = self._rbf(np.asarray(x_new, dtype=np.float64), self.x_all, self.params["length"])
        return Kn @ self.a[int(s_idx)]

    def predict(self, x_new):
        Kn = self._rbf(np.asarray(x_new, dtype=np.float64), self.x_all, self.params["length"])
        return np.einsum("nk,ik->in", Kn, self.a)

    # ---- KPCA（可視化・寄与率表示）----
    def fit(self):
        self.W0 = self.a.mean(axis=0)
        S = (self.a - self.W0[None, :]) @ self.K @ (self.a - self.W0[None, :]).T
        eVal, eVec = np.linalg.eigh(S)
        idx = np.argsort(-eVal)
        self.eVal = eVal[idx]
        self.eVec = eVec[:, idx]
        lam = self.eVal[:self.L]
        self.Z = self.eVec[:, :self.L] @ np.diag(np.sqrt(lam))
        self.W = np.diag(1.0 / lam) @ self.Z.T @ (self.a - self.W0[None, :])
        self.a_pos = self.Z @ self.W + self.W0

    def predict_approx(self, x_new):
        Kn = self._rbf(np.asarray(x_new, dtype=np.float64), self.x_all, self.params["length"])
        return np.einsum("nk,ik->in", Kn, self.a_pos)

    def generate_pos(self, x_new, Z):
        a = Z @ self.W + self.W0
        Kn = self._rbf(np.asarray(x_new, dtype=np.float64), self.x_all, self.params["length"])
        return np.einsum("nk,ik->in", Kn, a)
