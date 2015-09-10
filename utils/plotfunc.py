"""
Plot the graphs of some scalar function with 2d input variable.
"""
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from models.base import FuncModel
from models.testfunctions import TestFuncOne


# class FuncParam(object):
#     def __init__(self, A, b, c):
#         self.A_ = np.mat(A)
#         self.b_ = np.mat(b)
#         self.c_ = float(c)
#         self.grid_z = None
#         self.grid_x = None
#         self.grid_y = None


class FuncGraph(object):
    def __init__(self, func_model):
        if not isinstance(func_model, FuncModel):
            raise TypeError("Parameter func_mode must be class FuncModel.")
        else:
            self.model_ = func_model

    def _gen_variable(self, x_margin, y_margin, size=500):
        x = np.linspace(x_margin[0], x_margin[1], size)
        y = np.linspace(y_margin[0], y_margin[1], size)
        x, y = np.meshgrid(x, y)
        return x, y

    def _gen_target(self, grid_x, grid_y):
        grid_z = np.zeros(shape=grid_x.shape)
        grid_shape = grid_x.shape
        for i in range(0, grid_shape[0]):
            for j in range(0, grid_shape[1]):
                variable = [grid_x[i, j], grid_y[i, j]]
                value = self.model_.func_value(variable)
                grid_z[i, j] = value
        return grid_z

    def set_random_points(self, x_margin=(), y_margin=(), size=100):
        """
        Randomly set points which is used to plot the function graph.
        """
        try:
            grid_x, grid_y = self._gen_variable(x_margin, y_margin, size)
            grid_z = self._gen_target(grid_x, grid_y)
            self.grid_x = grid_x
            self.grid_y = grid_y
            self.grid_z = grid_z
            return True
        except:
            warn("Points does not initiate successfully.")
            return False

    def surface_graph(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim(0, 10)
        ax.plot_surface(self.grid_x, self.grid_y, self.grid_z, rstride=10, cstride=10, alpha=0.3)
        ax.contour(self.grid_x, self.grid_y, self.grid_z, zdir='z', offset=0, cmap=cm.coolwarm)
        return fig

    def contour_graph(self):
        fig = plt.figure()
        plt.contour(self.grid_x, self.grid_y, self.grid_z, zdir='z', offset=0)
        return fig


def main():
    """
    Python's main function to test this py document
    :return: None
    """
    # raise NotImplementedError("This branch hasn't been implemented yet!")
    a = np.array([1, 3, -0.1], dtype=np.float64)
    b = np.array([1, -3, -0.1], dtype=np.float64)
    c = np.array([-1, 0, -0.1], dtype=np.float64)
    name = "test_function_one"
    test_function_one = TestFuncOne(name=name, a=a, b=b, c=c)
    vi_func = FuncGraph(func_model=test_function_one)
    vi_func.set_random_points([-1.9, 0.49], [-0.47, 0.47])
    vi_func.surface_graph()
    vi_func.contour_graph()
    return


if __name__ == '__main__':
    main()
