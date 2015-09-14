"""
A non-quadratic test function.
"""
import numpy as np

from models.base import FuncModel


class TestFuncOne(FuncModel):
    """
    Test function one: y = exp(x_1 + 3x_2 - 0.1) + exp(x_1 - 3x_2 - 0.1) + exp(-x_1 - 0.1)
    """
    def __init__(self, name, a, b, c):
        super(FuncModel, self).__init__()
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and isinstance(c, np.ndarray):
            self.a = a
            self.b = b
            self.c = c
        else:
            raise TypeError("All the input parameter must be array or mat.")
        self.name_ = name

    def gradient(self, variable):
        x = np.append(variable, np.array([1]))
        a = np.array(self.a[0:2])
        b = np.array(self.b[0:2])
        c = np.array(self.c[0:2])
        g = a * np.exp(np.dot(self.a, x.T)) + b * np.exp(np.dot(self.b, x.T)) + c * np.exp(np.dot(self.c, x.T))
        return g

    def hessian(self, variable):
        x = np.append(variable, np.array([1]))
        a = np.mat(self.a[0:2])
        b = np.mat(self.b[0:2])
        c = np.mat(self.c[0:2])
        h = np.dot(a.T, a) * np.exp(np.dot(self.a, x.T)) + np.dot(b.T, b) * np.exp(np.dot(self.b, x.T)) \
            + np.dot(c.T, c) * np.exp(np.dot(self.c, x.T))
        return h

    def func_value(self, variable):
        x = np.append(variable, np.array([1]))
        y = np.exp(np.dot(self.a, x.T)) + np.exp(np.dot(self.b, x.T)) + np.exp(np.dot(self.c, x.T))
        return y

    def get_name(self):
        return self.name_


class TestFuncTwo(FuncModel):
    """
    A function in R^100. f(x) = c.T*x - sum_i(log(b_i - a_i.T * x), where the default vector dimension is 100 and m is
    500.
    """
    function_name = 'R100'
    def __init__(self, **kwargs):
        super(FuncModel, self).__init__()
        self.__dict__.update(kwargs)
        if not hasattr(self, 'dim'):
            self.dim = 100
        if not hasattr(self, 'm'):
            self.m = 500
        if not hasattr(self, 'c'):
            self.c = np.random.random(self.dim)
        if not hasattr(self, 'b'):
            self.b = np.random.random(self.m)
        if not hasattr(self, 'a'):
            self.a = np.random.random_sample(size=(self.m, self.dim))

    def gradient(self, variable):
        """
        Absolute function that returns the gradient of test function R100, with which some descent, newton method for
        example, can compute its step direction at each iteration.
        """
        x = variable
        g_left = self.c
        # self.b[0] - np.dot(self.a[1, :], x.T)
        g_right = [self.a[i, :] / (self.b[i] - np.dot(self.a[i, :], x.T)) for i in range(0, self.m)]
        g = g_left + np.sum(g_right, axis=0)
        return g

    def hessian(self, variable):
        x = variable
        h = np.zeros(shape=(self.dim, self.dim))
        for i in range(0, self.m):
            h -= np.dot(self.a[i, :].T, self.a[i, :]) / np.math.pow(self.b[i] - np.dot(self.a[i, :], x.T), 2)
        return h

    def func_value(self, variable):
        x = variable
        f = np.dot(self.c, x.T)
        for i in range(0, self.m):
            f -= np.log(self.b[i] - np.dot(self.a[i, :], x.T))
        return f

    def get_name(self):
        return self.function_name

    def get_func_param(self):
        func_param = {'dimension': self.dim, 'm': self.m, 'a': self.a, 'b': self.b, 'c':self.c}
        return dict(func_param)

def main_func_one():
    a = np.array([1, 3, -0.1], dtype=np.float64)
    b = np.array([1, -3, -0.1], dtype=np.float64)
    c = np.array([-1, 0, -0.1], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    name = "test_function_one"
    test_function_one = TestFuncOne(name=name, a=a, b=b, c=c)

    print test_function_one.func_value(x)
    print test_function_one.gradient(x)
    print test_function_one.hessian(x)


def main_func_two():
    """
    Main function which is used to test class TestFuncTwo.
    """
    a = np.mat(np.ones(shape=(500, 100)))
    b = np.mat(np.ones(shape=(500, 1)))
    c = np.mat(np.ones(shape=(1, 100)))
    test_func_R100 = TestFuncTwo(a=a, b=b, c=c)
    # print test_func_R100.get_func_param()
    x = np.mat(np.zeros(shape=(1, 100)))
    x_gradient = test_func_R100.hessian(x)
    x_func_value = test_func_R100.func_value(x)
    print "x_hessian: ", x_gradient
    print "x_hessian_size", x_gradient.shape
    print "x_func_value", x_func_value[0, 0]


if __name__ == '__main__':
    main_func_two()
    