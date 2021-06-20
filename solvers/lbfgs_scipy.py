from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.utils.stream_redirection import SuppressStd

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import fmin_l_bfgs_b


class Solver(BaseSolver):
    name = "scipy L-BFGS"

    install_cmd = "conda"
    requirements = ["scipy"]

    def set_objective(self, X, y, reg):
        self.X, self.y = X, y
        self.reg = reg

    def run(self, n_iter):
        _, n = self.X.shape

        x0 = np.zeros((n,))

        def func(w):
            residual = self.X.dot(w) - self.y
            f = 0.5 * residual.dot(residual) + 0.5 * self.reg * w.dot(w)
            grad = self.X.T @ residual + self.reg * w
            return f, grad

        bounds = [(0, np.inf)] * n

        out = SuppressStd()
        try:
            self.w, _, _ = fmin_l_bfgs_b(
                func, x0, bounds=bounds, pgtol=0.0, factr=0.0, maxiter=n_iter
            )
        except BaseException:
            print(out.output)
            raise

    def get_result(self):
        return self.w
