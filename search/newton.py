"""
Newton search algorithm.

A powerful method for unconstrained minimization problem.
"""
import sys
import numpy as np
from numpy.linalg import LinAlgError

from models.quadratic import FuncQuadratic
from models.base import FuncModel


class NewtonMethod(object):
    _model = None

    def __init__(self, model, dim, A, b, c):
        if not isinstance(model, FuncModel):
            raise TypeError("Type Error. %s is not a instance of class FuncModel." % model)
        else:
            self._model = model

        self.dim_ = dim
        self.A_ = np.mat(A, dtype=float)
        self.b_ = np.mat(b, dtype=float)
        self.c_ = c
        self.max_iter_ = sys.maxint
        self.tol_ = 0
        self.method_ = None

    def _newton_step(self, variable):
        """
        A descent direction called Newton step defined as delta(x_nt) = -invHessian(x)*Gradient(x)
        """
        x = np.mat(variable)
        gradient = self._model.gradient(x)
        hessian = self._model.hessian(x)
        try:
            inv_hessian = np.linalg.inv(hessian)
        except LinAlgError:
            # noinspection PyTypeChecker
            hessian = hessian + np.dot(np.eye(self.dim_), 1e-6)
            inv_hessian = np.linalg.inv(hessian)

        # Newton step
        step = -np.dot(inv_hessian, gradient.T).T

        return step

    def _stop_error(self, variable, newton_step):
        """
        Use Newton decrement as stopping as stopping error.
        lambda^2 = gradient(x) * hessian(x) * gradient(x)
        """
        g = self._model.gradient(variable)
        newton_decrement = np.dot(g, newton_step.T)
        return newton_decrement

    def set_max_iter(self, max_iter):
        if max_iter >= 1:
            self.max_iter_ = max_iter
            return True
        else:
            return False

    def set_tol(self, tol):
        if tol >= 0:
            self.tol_ = tol
            return True
        else:
            return False

    def set_search_method(self, method='linear_search'):
        self.method_ = method

    def base_newton_search(self):
        """
        Base Newton Method which use Newton step: - inv_Hessian(x) * Gradient(x).T as the search step.
        """
        count = 0
        error = sys.float_info.max
        x = np.random.random(self.dim_)
        while error > self.tol_ and count < self.max_iter_:
            newton_step = self._newton_step(x)
            x = x + newton_step
            error = self._stop_error(x, newton_step)
            count += 1

        return x

    def search(self):
        """
        Find the minimal solution of the given Quadratic function.
        """
        count = 0
        error = sys.float_info.max
        x = np.random.random(self.dim_)

        # Newton search
        while error > self.tol_ and count < self.max_iter_:
            newton_step = self._newton_step(x)
            alpha = -1
            beta = 1
            lambda_ = alpha + 0.383 * (beta - alpha)
            mu = alpha + 0.618 * (beta - alpha)
            # Linear search
            while np.math.fabs(beta - alpha) > 1e-10:
                # print 'search_lambda', alpha_k
                f_lambda = self._model.func_value(x + lambda_ * newton_step)
                f_mu = self._model.func_value(x + mu * newton_step)
                if f_lambda > f_mu:
                    alpha = lambda_
                    beta = beta
                    lambda_ = mu
                    mu = alpha + 0.618 * (beta - alpha)
                else:
                    alpha = alpha
                    beta = mu
                    mu = lambda_
                    lambda_ = alpha + 0.382 * (beta - alpha)
            # Final search direction
            x += alpha * newton_step
            count += 1
            error = self._stop_error(x, newton_step)
            print 'error: ', error
        # Optimal solution x
        return x


def main():
    A = np.mat([[1, 0], [0, 1]])
    b = np.mat([1, 1])
    c = 1.0
    # x = np.array([1, 1])

    test_func = FuncQuadratic(name="quadratic", A=A, b=b, c=c)
    a = NewtonMethod(dim=2, model=test_func, A=[[8.0, 0.0], [0.0, 8.0]], b=[0, 2], c=1)
    a.set_max_iter(500)
    a.set_tol(1e-4)
    x = a.base_newton_search()
    print x
    print "gradient:\n", np.dot(np.mat([[8, 0], [1, 8]]), x.T) + np.mat([0, 2]).T


if __name__ == '__main__':
    main()