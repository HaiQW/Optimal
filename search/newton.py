"""
Newton search algorithm.
"""
import sys
import numpy as np

from models.quadratic import FuncQuadratic
from models.base import FuncModel

class NewtonSearch(object):
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

    def gradient(self, x_curr):
        x = np.mat(x_curr)
        g = np.dot(x, self.A_) + self.b_
        return g

    def hessian(self):
        return self.A_

    def func_value(self, x_curr):
        """
        Calculate y of the Quadratic function at current value 'x_curr'
        :param x_curr: row vector
        :return: float
        """
        x = np.mat(x_curr)
        y = np.dot(x, np.dot(self.A_.T, x.T)) * 0.5 + np.dot(self.b_, x.T) + self.c_
        return y

    def search(self):
        """
        Find the minimal solution of the given Quadratic function.
        :return: row vector
        """
        count = 0
        error = sys.float_info.max
        x = np.random.random(self.dim_) * 10

        # Newton search
        while error > self.tol_ and count < self.max_iter_:
            # Calculate the search direction
            # inv_hessian = np.linalg.inv(self.hessian())
            inv_hessian = np.linalg.inv(self._model.hessian(x))
            # search_dir = np.dot(self.gradient(x), inv_hessian)
            search_dir = np.dot(self._model.gradient(x), inv_hessian)
            alpha_k = -0.5
            beta_k = 0.5
            lambda_k = alpha_k + 0.383 * (beta_k - alpha_k)
            mu_k = alpha_k + 0.618 * (beta_k - alpha_k)
            # Linear search to find the optimal step length at the search
            # direction 'search_dir'
            while np.math.fabs(beta_k - alpha_k) > 1e-10:
                # print 'search_lambda', alpha_k
                # f_lambda_k = self.func_value(x + lambda_k * search_dir)
                # f_mu_k = self.func_value(x + mu_k * search_dir)
                # print f_lambda_k, f_mu_k
                f_lambda_k = self._model.func_value(x + lambda_k * search_dir)
                f_mu_k = self._model.func_value(x + mu_k * search_dir)
                if f_lambda_k > f_mu_k:
                    alpha_k = lambda_k
                    beta_k = beta_k
                    lambda_k = mu_k
                    mu_k = alpha_k + 0.618 * (beta_k - alpha_k)
                else:
                    alpha_k = alpha_k
                    beta_k = mu_k
                    mu_k = lambda_k
                    lambda_k = alpha_k + 0.382 * (beta_k - alpha_k)

            x = x + alpha_k * search_dir
            count += 1
            # error = np.math.sqrt(np.dot(self.gradient(x), self.gradient(x).T))
            error = np.math.sqrt(np.dot(self._model.gradient(x), self._model.gradient(x).T))
            print 'error: ', error

        return x


def main():
    A = np.mat([[1, 0], [0, 1]])
    b = np.mat([1, 1])
    c = 1.0
    x = np.array([1, 1])

    test_func = FuncQuadratic(name="quadratic", A=A, b=b, c=c)
    a = NewtonSearch(dim=2, model= test_func, A=[[8.0, 4.0], [2.0, 8.0]], b=[0, 2], c=1)
    a.set_max_iter(500)
    a.set_tol(1e-4)
    x = a.search()
    print x
    print 'gradient:\n', np.dot(np.mat([[8, 0], [1, 8]]), x.T) + np.mat([0, 2]).T


if __name__ == '__main__':
    main()