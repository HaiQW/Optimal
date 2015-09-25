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


def pre_plot(data_set, data_tag, attr=None):
    if attr == None:
        print "Error! Please give attribute name of the input data set."
        return -1
    sns.set(style="ticks", color_codes=True)
    target = np.reshape(data_tag, newshape=(data_tag.size, 1))
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

    # Config parameters
    rare_index = 2
    rare_size = 50
    rare_sample_size = 2
    major_index = 5
    major_sample_size = 10
    major_size = 200
    dim = 3

    print "Loading synthetic data set..."
    source = np.loadtxt("../SyntheticData/Synthetic_data.txt")
    data_set = source[:, 0:dim]
    data_tag = source[:, -1]

    print "Pre_processing data set to form the RCE scenario..."
    rare_sample_data, rare_sample_tag = get_seeds(data_set, data_tag, rare_index, rare_sample_size)
    major_sample_data, major_sample_tag = get_seeds(data_set, data_tag, major_index, major_sample_size)

    # plot seed examples from rare category and major category
    data = np.append(major_sample_data, rare_sample_data, axis=0)
    tag = np.append(major_sample_tag, rare_sample_tag, axis=0)
    f = pre_plot(data_set=data, data_tag=tag, attr=["X", "Y", "Z", "category"])
    plt.show(f)

    # Config searching parameter
    # rare_test = np.array([[ 4.50483295, -4.43064881], [4.60141853, -4.19811788]])
    # major_test = np.array([[-1.78122071, 7.74226893], [2.69124233, 3.29862953]])
    init = np.ones(shape=(1, 3))
    rare_func = RareFunc(name="rare", rare=rare_sample_data, major=major_sample_data, c=0.01, dim=dim)
    test_gradient = GradientDescentMethod(rare_func, stop_error=1e-6, max_iter=200, init_point=init)
    # print "gradient", rare_func.gradient(init)

    print "Search the diagonal matrix (A)..."
    x = test_gradient.search('exact')


    # Regression test
    projection = np.array([np.math.sqrt(x[0, 0]), np.math.sqrt(x[0, 1]), np.math.sqrt(x[0, 2])])
    projection = np.diag(projection)
    rare_data, rare_tag = get_seeds(data_set, data_tag, rare_index, rare_size)
    major_data, major_tag = get_seeds(data_set, data_tag, major_index, major_size)
    p_rare_data = np.dot(rare_data, projection)
    p_major_data = np.dot(major_data, projection)
    p_data = np.append(p_major_data, p_rare_data, axis=0)
    p_tag = np.append(major_tag, rare_tag, axis=0)
    f = pre_plot(data_set=p_data, data_tag=p_tag, attr=["X", "Y", "Z", "category"])
    plt.show(f)
    # rare_func.gradient(init)
    print "rare_gradient_a:\n", rare_func.gradient(x)
    print x

if __name__ == "__main__":
    main()

