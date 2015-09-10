"""
Some useful test function classes.
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


def main():
    a = np.array([1, 3, -0.1], dtype=np.float64)
    b = np.array([1, -3, -0.1], dtype=np.float64)
    c = np.array([-1, 0, -0.1], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    name = "test_function_one"
    test_function_one = TestFuncOne(name=name, a=a, b=b, c=c)

    print test_function_one.func_value(x)
    print test_function_one.gradient(x)
    print test_function_one.hessian(x)


if __name__ == '__main__':
    main()
    