"""
The interface modules.
"""
import numpy as np

from abc import ABCMeta, abstractmethod


class FuncModel(object):
    """
    A abstract class is used to provide interface of functions, for example
    the gradient function and hessian function which are needed in Newton
    method for solving optimization problem.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.func_name_ = "Abstract"

    @abstractmethod
    def gradient(self, variable):
        pass

    @abstractmethod
    def hessian(self, variable):
        pass

    @abstractmethod
    def func_value(self, variable):
        pass

    @abstractmethod
    def get_name(self):
        pass