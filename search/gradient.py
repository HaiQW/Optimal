"""
Gradient descent method, or gradient method with backtracking line search,
with a = 0.1, b = 0.1
"""
import numpy as np
import sys
from copy import deepcopy

from models.base import FuncModel
from models.testfunctions import TestFuncOne
from models.testfunctions import TestFuncTwo
from utils.plotfunc import FuncGraph


class GradientDescentMethod(object):
    max_iter = 100
    stop_error = 1e-6
    dim = 2
    iter_points = []
    iter_errors = []
    iter_counts = 0

    def __init__(self, func_model, **kwargs):
        if not isinstance(func_model, FuncModel):
            raise TypeError("TypeError. Parameter func_model must be FuncModel.")
        else:
            self.model = func_model
        self.__dict__.update(**kwargs)
        if not hasattr(self, 'init_point'):
            self.init_point = np.random.random(self.dim)

    def _calc_search_direction(self, variable):
        """
        Calculate the search direction of gradient descent method, which is simply the
        negative gradient direction at each point in dom(f).
        """
        g = -1 * self.model.gradient(variable)
        return g

    def _backtracking_line_search(self, curr_point, search_dir, alpha=0.1, beta=0.5):
        """
        A very simple and effective inexact line search methods called backtracking,
        which depends on two constants alpha, beta, with 0 < alpha < 0.5, 0 < beta < 1.
        """
        x = curr_point
        delta_x = search_dir
        diff_fx = self.model.gradient(x)
        fx = self.model.func_value(x)
        a = alpha
        b = beta
        t = 1
        # print "x", x
        # print delta_x
        # print fx + a * t * np.dot(diff_fx, delta_x.T)
        while self.model.func_value(x + t * delta_x) > (fx + a * t * np.dot(diff_fx, delta_x.T)):
            t *= b
        return t

    def _exact_line_search(self, curr_point, search_dir):
        """
        A line search method that chooses t to minimize f along the ray {x + t * delta_x | t >= 0} as follows.
        t = arg_min f(x + t * delta_x). This exact line search is used when the cost of minimization problem
        with one variable is low compared to the cost of computing the search direction itself.
        """
        granularity = 1.0 / 100.0
        t = np.array(np.linspace(0, 99, 100))
        t = t * granularity
        fs = [self.model.func_value(curr_point + s * search_dir) for s in t]
        index = np.argmin(fs)
        return index * granularity

    def _calc_error(self, point):
        x = point
        g = self.model.gradient(x)
        e = np.math.sqrt(np.dot(g, g.T))
        return e

    def set_init_point(self, point):
        if len(point) != self.dim:
            # raise TypeError("TypeError. Initial point must have length of %s." % self.dim)
            return False
        else:
            self.init_point = np.array(point)
            return True

    def set_max_iter(self, max_iter):
        if max_iter < 0:
            return False
        else:
            self.max_iter = max_iter
            return True

    def set_stop_error(self, stop_error):
        if stop_error <= 0:
            return False
        else:
            self.stop_error = stop_error
            return True

    def get_iter_points(self):
        return self.iter_points

    def get_iter_errors(self):
        return self.iter_errors

    def get_iter_counts(self):
        return self.iter_counts

    def search(self, method):
        """
        Gradient descent method with backtracking line search.
        """
        x = self.init_point
        error = self._calc_error(x)
        count = 0
        self.iter_points.append(np.array(x))  # Save points of each iteration
        self.iter_errors.append(error)
        while error > self.stop_error and count < self.max_iter:
            search_direction = self._calc_search_direction(x)
            if method == 'backtracking':
                t = self._backtracking_line_search(x, search_direction)
            elif method == 'exact':
                t = self._exact_line_search(x, search_direction)
            else:
                raise TypeError("No such a searching method, got %r" % method)
            x += t * search_direction
            error = self._calc_error(x)
            self.iter_errors.append(error)
            self.iter_points.append(deepcopy(x))
            count += 1
        self.iter_counts = count
        return x


def main():
    a = np.array([1, 3, -0.1], dtype=np.float64)
    b = np.array([1, -3, -0.1], dtype=np.float64)
    c = np.array([-1, 0, -0.1], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    name = "test_function_one"
    model = TestFuncOne(name=name, a=a, b=b, c=c)
    vi_func = FuncGraph(func_model=model)
    vi_func.set_random_points([-1.9, 0.49], [-0.47, 0.47])
    func_graph = vi_func.contour_graph()
    test_gradient = GradientDescentMethod(model, stop_error=1e-6, max_iter=200, init_point=[-0.7, 0.45])
    var = test_gradient.max_iter
    # x_optimal = test_gradient.search()
    # print test_gradient.init_point
    optimal = test_gradient.search('exact')
    iter_points = np.array(test_gradient.get_iter_points())
    import matplotlib.pyplot as plt
    plt.plot(iter_points[:, 0], iter_points[:, 1], marker="o")
    plt.show()
    print iter_points


def test_func_two():
    """
    Main function which is used to test class TestFuncTwo.
    """
    a = np.mat(np.ones(shape=(500, 100)))
    b = np.mat(np.ones(shape=(500, 1)))
    c = np.mat(np.ones(shape=(1, 100)))
    test_func_R100 = TestFuncTwo(a=a, b=b, c=c)
    x = np.mat(np.zeros(shape=(1, 100)))
    test_gradient = GradientDescentMethod(test_func_R100, stop_error=1e-6, max_iter=200, init_point=x)
    optimal = test_gradient.search('backtracking')
    iter_points = np.array(test_gradient.get_iter_points())
    iter_errors = np.array(test_gradient.get_iter_errors())
    iter_counts = np.array(test_gradient.get_iter_counts())
    print iter_errors


if __name__ == '__main__':
    test_func_two()
