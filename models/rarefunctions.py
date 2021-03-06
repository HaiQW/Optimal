"""
A function to minimize the sum of distance with similar constraints
"""
import numpy as np
from models.base import FuncModel


class RareFunc(FuncModel):
    """
    Test function one: y = exp(x_1 + 3x_2 - 0.1) + exp(x_1 - 3x_2 - 0.1) + exp(-x_1 - 0.1)
    """
    distance_help = 0.0
    gradient_help = None
    hessian_help = None

    def __init__(self, name, rare, major, c, dim):
        super(FuncModel, self).__init__()
        if isinstance(rare, np.ndarray) and isinstance(major, np.ndarray):
            self.rare = rare
            self.major = major
            self.dim = dim
            self.c = c
            self.major_size = major.shape[0]
            self.rare_size = rare.shape[0]
            self.gradient_help = self._init()
        else:
            raise TypeError("All the input parameter must be array or mat.")
        self.name_ = name


    def _init(self):
        derive_sum_const = np.zeros(shape=(1, self.dim))
        for i in range(0, self.major_size):
            for j in range(0, i):
                d = np.array(self.major[i, :] - self.major[j, :])
                derive_sum_const += d * d

        # similar pairwise constraints in major category
        for i in range(0, self.rare_size):
            for j in range(0, i):
                d = np.array(self.rare[i, :] - self.rare[j, :])
                derive_sum_const += d * d

        return derive_sum_const


    def gradient(self, variable):
        x = np.array(variable)
        if not (x > 0).all():
            # raise ValueError("ValueError. dia(A) must greater than zero vector, got negative values.")
            x = np.zeros(shape=(1, self.dim))
        g = np.zeros(shape=(1, self.dim))
        h = np.zeros(shape=(self.dim, self.dim))
        sum_ = 0.0

        # gradient in dissimilar pairwise constraints
        for i in range(0, self.rare_size):
            for j in range(0, self.major_size):
                d_ij = self.rare[i, :] - self.major[j, :]
                dis = np.sqrt(np.dot(d_ij * d_ij, x.T))
                g += 0.5 * d_ij * d_ij / (((dis == 0) * 1e-6) + dis)
                h -= 0.25 * np.dot((d_ij * d_ij).T, d_ij * d_ij) / (np.math.pow(dis, 3) + (dis == 0) * 1e-6)
                sum_ += dis
        g = self.gradient_help - (self.c * g / (sum_ + (sum_ == 0) * 1e-6))
        self.distance_help = sum_
        self.hessian_help = h
        return g

    def hessian(self, variable):
        # Calculate the gradient of the gradient

        h = self.hessian_help / self.distance_help -\
            np.dot(self.gradient_help.T, self.gradient_help)/np.math.pow(self.distance_help, 2)
        h = - self.c * h
        return h

    def func_value(self, variable):
        # f = np.dot(self.gradient_help, x.T) + self.c * np.math.log(self.distance_help)
        x = np.array(variable)
        if not (x > 0).all():
            x = np.ones(shape=(1, self.dim)) * 1e-6
        f = np.dot(self.gradient_help, x.T)

        # calculate distance
        sum_dis = 0
        for i in range(0, self.major.shape[0]):
            for j in range(0, self.rare.shape[0]):
                vec_ij = self.major[i, :] - self.rare[j, :]
                dis_ij = np.math.sqrt(np.dot(vec_ij * vec_ij, x.T))
                sum_dis += dis_ij
        f += self.c * np.math.log(sum_dis)
        return f

    def get_name(self):
        return self.name_
