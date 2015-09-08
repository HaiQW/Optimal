"""
Visualize some function.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from models.base import FuncModel
from models.testfunctions import TestFuncOne

class FuncParam(object):
    def __init__(self, A, b, c):
        self.A_ = np.mat(A)
        self.b_ = np.mat(b)
        self.c_ = float(c)


class ViFunc(object):
    def __init__(self, func_model):
        if not isinstance(func_model, FuncModel):
            raise TypeError("Parameter func_mode must be class FuncModel.")
        else:
            self.model_ = func_model

    def gen_variable(self, x_margin, y_margin, size = 500):
        x = np.linspace(x_margin[0], x_margin[1], size)
        y = np.linspace(y_margin[0], y_margin[1], size)
        x, y = np.meshgrid(x, y)
        return x, y

    def gen_target(self, grid_x, grid_y):
        z = np.zeros(shape=grid_x.shape)
        grid_shape = grid_x.shape
        for i in range(0, grid_shape[0]):
            for j in range(0, grid_shape[1]):
                variable = [grid_x[i, j], grid_y[i, j]]
                value = self.model_.func_value(variable)
                z[i, j] = value
        return z

    # def calc_func(self, variable):
    #     # assert isinstance(self.func_param_.b_, FuncParam)
    #     if not isinstance(self.func_param_, FuncParam):
    #         raise TypeError("Not a instance of Class FuncParam")
    #     else:
    #         value = np.dot(np.dot(variable, self.func_param_.A_), variable.T) * 0.5
    #         value += np.dot(self.func_param_.b_, variable.T) + self.func_param_.c_
    #     return value

    def plt_func(self):
        grid_x, grid_y = self.gen_variable([-1.8, 0.9], [-0.5, 0.5], size=400)
        grid_z = self.gen_target(grid_x, grid_y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim(0, 10)
        ax.plot_surface(grid_x, grid_y, grid_z, rstride=10, cstride=10, alpha=0.3)
        ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap=cm.coolwarm)
        plt.show(fig)


def main():
    """
    Python's main function to test this py document
    :return: None
    """
    # raise NotImplementedError("This branch hasn't been implemented yet!")
    a = np.array([1, 3, -0.1], dtype=np.float64)
    b = np.array([1, -3, -0.1], dtype=np.float64)
    c = np.array([-1, 0, -0.1], dtype=np.float64)
    x = np.array([1, 1], dtype=np.float64)
    name = "test_function_one"
    x = np.array([5, -5], dtype=np.float64)
    test_function_one = TestFuncOne(name=name, a=a, b=b, c=c)
    print test_function_one.func_value(x)
    vi_func = ViFunc(func_model=test_function_one)
    vi_func.plt_func()
    return


if __name__ == '__main__':
    main()