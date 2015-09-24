import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from models.rarefunctions import RareFunc
from search.newton import NewtonMethod
from search.gradient import GradientDescentMethod

def pre_process(data_set, target, rare_index, rare_size):
    """
    Pre_process the given "data_set" and get the target data set suit for
    RCE scenario.
    :return: Object(RCE_set, RCE_target)
    """
    data_set = np.array(data_set)
    target = np.array(target)

    # Get rare category data set and their labels
    rare_class = data_set[target == rare_index, :]
    np.random.shuffle(rare_class)
    rare_data = rare_class[0:rare_size, :]
    rare_target = np.ones(shape=(rare_size, 1)) * rare_index

    # Get the final data set and their labels
    final_data = np.append(data_set[(target != rare_index), :], rare_data, axis=0)
    final_target = np.append(target[(target != rare_index)], rare_target)

    return final_data, final_target


def m_distance_matrix():
    """
    Calculate Mahalanobis distance matrix.
    :return: matrix
    """
    pass


def get_seeds(data_set, data_target, seeds_category, seeds_size=1):
    """
    Randomly get some seeds from a given rare category.
    """
    if  (seeds_category not in data_target) \
            or (data_set.shape[0] != data_target.shape[0]):
        return [], []

    if seeds_size <= 0 or seeds_size >= len(data_set):
        seeds_size = 1

    # Select data examples belonging to the given 'seeds_category'
    category_set =  data_set[(data_target == seeds_category), :]
    np.random.shuffle(category_set)
    # Randomly get 'seeds_size' data example index
    seeds_data = category_set[0:seeds_size, :]
    seeds_target = np.ones(shape=(seeds_size, 1), dtype=int) * seeds_category

    return seeds_data, seeds_target


def gen_pairwise_constraints(major_data, rare_data):
    pass
    # """
    # Generate all pairwise constraints from the two categories, major and rare
    # """
    # major_numbers = major_data.shape[0]
    # rare_numbers = rare_data.shape[0]
    # constraints = np.ones(shape=(major_numbers, rare_numbers))
    # s_pairs = np.ones(shape=(int(major_numbers*major_numbers*0.5) + int(rare_numbers*rare_data*0.5), 2))
    # d_pairs = np.ones(shape=(major_numbers*rare_numbers, 2))
    #
    # # generate similar constraints
    # for i in range(0, major_numbers):
    #     for j in range(0, rare_numbers):
    #         s_pairs
    #
    # pass


def pre_plot(data_set, target, attr=None):
    if attr == None:
        print "Error! Please give attribute name of the input data set."
        return -1
    sns.set(style="ticks", color_codes=True)
    target = np.reshape(target, newshape=(target.size, 1))
    data_frame = pd.DataFrame(np.append(data_set, target, axis=1), columns=attr)
    print data_frame
    # Get pair_plot figure
    figure = sns.pairplot(data_frame,
                          vars=attr[0:-1],
                          hue=attr[-1])
    return figure


def main():
    """
    Main function of python.
    """
    print "Loading synthetic data set..."
    source = np.loadtxt("../SyntheticData/Synthetic_data.txt")
    data_set = source[:, 0:2]
    data_target = source[:, -1]

    print "Pre_processing data set to form the RCE scenario..."
    seeds_data, seeds_tag = get_seeds(data_set, data_target, 2, 2)
    major_data, major_tag = get_seeds(data_set, data_target, 5, 2)
    print seeds_data
    print major_data
    # f = pre_plot(data_set=major_data, target=major_tag, attr=["X", "Y", "category"])
    # plt.show(f)
    init = np.ones(shape=(1, 2))
    rare_test = np.array([[ 4.50483295, -4.43064881], [4.60141853, -4.19811788]])
    major_test = np.array([[-1.78122071, 7.74226893], [2.69124233, 3.29862953]])
    rare = RareFunc(name="rare", rare=rare_test, major=major_test, c=1, dim=2)
    test_gradient = GradientDescentMethod(rare, stop_error=1e-6, max_iter=200, init_point=init)
    print "gradient", rare.gradient(init)
    test_gradient.search('exact')
    rare.gradient(init)
    print "rare_gradient_a:\n", rare.hessian(init)


if __name__ == "__main__":
    main()

