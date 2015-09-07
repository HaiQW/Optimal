"""
Visualize some function.
"""
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d


class FuncParam(object):
    def __init__(self, A, b, c):
        self.A_ = np.mat(A)
        self.b_ = np.mat(b)
        self.c_ = float(c)

class FuncModel(object):
    pass

class ViFunc(object):
    def __init__(self, func_name, func_param):
        self.name_ = func_name
        self.func_param_ = func_param

    def gen_variable(self, margin_left=-100, margin_right=100, size = 500):
        x = np.linspace(margin_left, margin_right, size)
        y = np.linspace(margin_left, margin_right, size)
        x, y = np.meshgrid(x, y)
        return x, y

    def gen_target(self, grid_x, grid_y):
        z = np.zeros(shape=grid_x.shape)
        grid_shape = grid_x.shape
        for i in range(0, grid_shape[0]):
            for j in range(0, grid_shape[1]):
                variable = np.mat([grid_x[i, j], grid_y[i, j]])
                value = self.calc_func(variable)
                z[i, j] = value

        return z

    def calc_func(self, variable):
        # assert isinstance(self.func_param_.b_, FuncParam)
        if not isinstance(self.func_param_, FuncParam):
            raise TypeError("Not a instance of Class FuncParam")
        else:
            value = np.dot(np.dot(variable, self.func_param_.A_), variable.T) * 0.5
            value += np.dot(self.func_param_.b_, variable.T) + self.func_param_.c_
        return value

    def plt_func(self):
        grid_x, grid_y = self.gen_variable()
        grid_z = self.gen_target(grid_x, grid_y)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(grid_x, grid_y, grid_z, rstride=8, cstride=8, alpha=0.3)
        plt.show(fig)


def main():
    """
    Python's main function to test this py document
    :return: None
    """
    # raise NotImplementedError("This branch hasn't been implemented yet!")
    A = [[1, 0], [0, -1]]
    b = [2, 4]
    c = 5.0
    func_param = FuncParam(A, b, c)
    vi_func = ViFunc(func_name="quadratic", func_param=func_param)
    vi_func.plt_func()
    return


if __name__ == '__main__':
    main()