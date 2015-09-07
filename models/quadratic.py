"""
Quadratic function.
"""
import numpy as np

from models.base import FuncModel


class FuncQuadratic(FuncModel):
    func_name_ = None
    A_ = np.mat
    b_ = np.array
    c_ = None

    def __init__(self, name=None, A=None, b=None, c=None):
        FuncModel.__init__(self)

        if name is not None:
            self.func_name_ = name
        elif not setattr(self, 'name', None):
            raise ValueError("%s must have a name" % type(self).__name__)

        if A is None:
            self.A_ = np.mat([])
        elif isinstance(A, np.ndarray):
            self.A_ = A
        else:
            raise TypeError("A must be a matrix")

        if b is None:
            self.b_ = np.array([])
        elif isinstance(b, np.ndarray):
            self.b_ = b
        else:
            raise TypeError("b must be a matrix")

        if c is None:
            self.c_ = np.float64(0.0)
        elif isinstance(c, float) or isinstance(c, int):
            self.c_ = np.float64(c)
        else:
            raise TypeError("c must be a real number")

        self.func_name_ = name

    def gradient(self, variable):
        x = np.mat(variable)
        g = np.dot(self.A_, x.T) + self.b_.T
        g = g.T
        return g

    def hessian(self, variable):
        h = self.A_
        return h

    def func_value(self, variable):
        x = np.array(variable)
        value = np.dot(np.dot(x, self.A_), x.T) * 0.5 + np.dot(self.b_, x.T) + np.mat(self.c_)
        value = np.float64(value)
        return value

    def get_name(self):
        return self.func_name_


def main():
    """
    Main test function for this module named models.
    """
    A = np.mat([[1, 0], [0, 1]])
    b = np.mat([1, 1])
    c = 1.0
    x = np.array([1, 1])

    test_func = FuncQuadratic(name="quadratic", A=A, b=b, c=c)
    h = test_func.hessian(x)
    g = test_func.gradient(x)
    y = test_func.func_value(x)

    print "input variable x : ", x
    print "gradient of x: ", g
    print "hessian of x: ", h
    print "f(x): ", y


if __name__ == '__main__':
    main()
