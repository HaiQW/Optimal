import numpy as np

def func1(func_param):
    print "func1 %s" % func_param

def func2(fun_name, func_param):
    fun_name(func_param)

def main():
    func2(func1, "func1_param")

if __name__ == '__main__':
    main()