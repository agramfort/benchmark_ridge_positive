from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse, optimize


def _get_rescaled_operator(X, X_offset, X_scale):
    X_offset_scale = X_offset / X_scale

    def matvec(b):
        return X.dot(b) - b.dot(X_offset_scale)

    def rmatvec(b):
        return X.T.dot(b) - X_offset_scale * np.sum(b)

    X1 = sparse.linalg.LinearOperator(
        shape=X.shape, matvec=matvec, rmatvec=rmatvec)
    return X1


def _solve_trf(X, y, reg, max_iter, tol=0, X_offset=None, X_scale=None):
    lsq_config = {
        "method": "trf",
        "max_iter": max_iter,
        "tol": tol,
    }
    n_samples, n_features = X.shape

    if X_offset is None or X_scale is None:
        X1 = sparse.linalg.aslinearoperator(X)
    else:
        X1 = _get_rescaled_operator(X, X_offset, X_scale)

    Xa_shape = (n_samples + n_features, n_features)
    bounds = (0, np.inf)
    sqrt_alpha = np.sqrt(reg)

    def mv(b):
        return np.hstack([X1.matvec(b), sqrt_alpha * b])

    def rmv(b):
        return X1.rmatvec(b[:n_samples]) + sqrt_alpha * b[n_samples:]

    Xa = sparse.linalg.LinearOperator(shape=Xa_shape, matvec=mv, rmatvec=rmv)
    y_zeros = np.zeros(n_features, dtype=X.dtype)
    y_column = np.hstack([y, y_zeros])
    result = optimize.lsq_linear(Xa, y_column, bounds=bounds, **lsq_config)

    return result["x"]


class Solver(BaseSolver):
    name = "scipy TRF"

    install_cmd = "conda"
    requirements = ["scipy"]

    def set_objective(self, X, y, reg):
        self.X, self.y = X, y
        self.reg = reg

    def run(self, n_iter):
        self.w = _solve_trf(
            self.X,
            self.y,
            self.reg,
            max_iter=n_iter + 1,
            tol=0,
            X_offset=None,
            X_scale=None,
        )

    def get_result(self):
        return self.w
