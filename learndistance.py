"""
Rare category exploration based on the side information.
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.datasets as database
import sklearn.cluster as cluster
from models.rarefunctions import RareFunc
from search.gradient import GradientDescentMethod
from sklearn import svm # Outlier detection model
from sklearn.neighbors import NearestNeighbors

class RareParam:
    def __init__(self, data_set, tag_set, size, dim, rare_index, major_index, rare_sample_size, major_sample_size):
        self.data_set = data_set
        self.tag_set = tag_set
        self.rare_index = rare_index
        self.major_index = major_index
        self.size = size
        self.dim = dim
        self.rare_sample_size = rare_sample_size
        self.major_sample_size = major_sample_size
        self.diagonal_a = None

class _rec_help:
    """Help create rare category exploration scenario.
    """
    def __init__(self):
        pass

class RareCategoryExploration:
    """Rare category exploration
    :parameter
    ----------
    max_iter: int, default: 300

    :attribute
    ----------
    indices_: indices of data examples from the objective rare category
    """
    def __init__(self, param):
        if not isinstance(param, RareParam):
            raise ValueError("RareCategory.__init__(self, param) takes RareParam.")
        else:
            self.param = param

        # Generate sample data
        self.rare_sample_data, self.rare_sample_tag = self._get_seeds(self.param.data_set, self.param.tag_set,
                                                                      self.param.rare_index,
                                                                      self.param.rare_sample_size)
        self.major_sample_data, self.major_sample_tag = self._get_seeds(self.param.data_set, self.param.tag_set,
                                                                        self.param.major_index,
                                                                        self.param.major_sample_size)

    def _check_fit_data(self, X):
        """Verify that the number of data examples given is larger than k"""

        pass


    def fit(self, data_set):
        pass

    @staticmethod
    def _pre_process(data_set, tag_set, rare_index, rare_size):
        """
        Pre_process the given "data_set" and get the target data set suit for RCE scenario.
        """
        data_set = np.array(data_set)
        target = np.array(tag_set)

        # Get rare category data set and their labels
        rare_class = data_set[target == rare_index, :]
        np.random.shuffle(rare_class)
        rare_data = rare_class[0:rare_size, :]
        rare_target = np.ones(shape=(rare_size, 1)) * rare_index

        # Get the final data set and their labels
        final_data = np.append(data_set[(target != rare_index), :], rare_data, axis=0)
        final_target = np.append(target[(target != rare_index)], rare_target)

        return final_data, final_target

    @staticmethod
    def _get_seeds(data_set, tag_set, seeds_category, seeds_size=1):
        """
        Randomly get some seeds from a given rare category.
        """
        if (seeds_category not in tag_set) \
                or (data_set.shape[0] != tag_set.shape[0]):
            return [], []

        if seeds_size <= 0 or seeds_size >= len(data_set):
            seeds_size = 1

        # Select data examples belonging to the given 'seeds_category'
        category_set = data_set[(tag_set == seeds_category), :]
        np.random.shuffle(category_set)
        # Randomly get 'seeds_size' data example index
        seeds_data = category_set[0:seeds_size, :]
        seeds_target = np.ones(shape=(seeds_size, 1), dtype=int) * seeds_category

        return seeds_data, seeds_target

    @staticmethod
    def _pre_plot(data_set, tag_set, attr=None):
        if not attr:
            print "Error! Please give attribute name of the input data set."
            return None
        sns.set(style="ticks", color_codes=True)
        target = np.reshape(tag_set, newshape=(tag_set.size, 1))
        data_frame = pd.DataFrame(np.append(data_set, target, axis=1), columns=attr)
        # Get pair_plot figure
        figure = sns.pairplot(data_frame, vars=attr[0:-1], hue=attr[-1])
        return figure

    @staticmethod
    def _pre_plot_3d(data_set):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_set[0:500, 0], data_set[0:500, 2], data_set[0:500, 1], marker="o", c="blue", s=20, lw=0.001,
                   label="major")
        ax.scatter(data_set[500:-1, 0], data_set[500:-1, 2], data_set[500:-1, 1], marker=">", s=20, c="red", lw=0.001,
                   label="rare")
        z_lim = ax.get_zlim()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.set_zticks(np.linspace(z_lim[0], z_lim[1], 3))
        ax.set_xticks(np.linspace(x_lim[0], x_lim[1], 3))
        ax.set_yticks(np.linspace(y_lim[0], y_lim[1], 3))
        return fig

    def get_rescaled_data(self):
        """
        Get the data set, which is rescaled by the learned diagonal A.
        """
        x = self.param.diagonal_a[0, :]
        p_matrix = np.diag([np.sqrt(x[i]) for i in range(0, self.param.dim)])  # the projection matrix
        major_data = self.param.data_set[self.param.tag_set == self.param.major_index, :]
        rare_data = self.param.data_set[self.param.tag_set == self.param.rare_index, :]
        p_rare_data = np.dot(rare_data, p_matrix)
        p_major_data = np.dot(major_data, p_matrix)
        p_data = np.append(p_major_data, p_rare_data, axis=0)
        # p_data = np.reshape(p_data, newshape=(self.param.size, self.param.dim))
        return p_data
        # f = self._pre_plot_3d(data_set=p_data)
        # plt.show(f)
        # f.savefig('/home/haiqw/Dropbox/Latex/Mahalanobis/pic/mahalanobis.eps', format='eps', bbox_inches='tight',
        #           facecolor='w', edgecolor='w', transparent=True, dpi=1200)

    def exploration(self, name="rare", c=0.1, stop_error=1e-6, max_iter=100, init_point=None):
        if not init_point:
            init_point = np.ones(shape=(1, self.param.dim)) * 5
        rare_func = RareFunc(name=name, rare=self.rare_sample_data, major=self.major_sample_data, c=c,
                             dim=self.param.dim)
        test_gradient = GradientDescentMethod(rare_func, stop_error=stop_error, max_iter=max_iter,
                                              init_point=init_point)
        # calculate diagonal distance metrix x
        x = test_gradient.search('exact')

        # find the all data examples from the given rare category
        self.param.diagonal_a = x
        rescaled_data = self.get_rescaled_data()
        distance_matrix = np.zeros(shape=(1, self.param.size))
        for j in range(0, self.param.rare_sample_size):
            seed = self.rare_sample_data[j, :]
            distance_matrix += np.array([np.dot(self.param.data_set[i, :] - seed, self.param.data_set[i, :] - seed) for i in range(0, self.param.size)])

        # Use K-means to find the coarse shaped cluster, which contains data examples from rare category.
        init_point = self.rare_sample_data
        k_means = cluster.KMeans(n_clusters=4, max_iter=500)
        k_means.fit(rescaled_data)
        pre_tag = np.array(k_means.predict(rescaled_data))
        coarse_index = np.array(np.argwhere(pre_tag == pre_tag[-1]))
        coarse_data_set = rescaled_data[pre_tag == pre_tag[-1], :]

        # Use nearest-neighbors
        x = self.param.diagonal_a[0, :]
        p_matrix = np.diag([np.sqrt(x[i]) for i in range(0, self.param.dim)])
        seed = np.dot(self.rare_sample_data, p_matrix)
        neighbor_model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
        neighbor_model.fit(rescaled_data)
        distance, indices = neighbor_model.kneighbors(np.average(seed, axis=0))
        print indices

        # Remove outliers
        coarse_data_set = rescaled_data[indices[0, :], :]
        coarse_index = indices[0, :]
        model_outlier = svm.OneClassSVM(nu=0.95 * 0.3 + 0.05, kernel="rbf", gamma=0.)
        model_outlier.fit(coarse_data_set)
        outlier_tag = model_outlier.predict(coarse_data_set)
        refined_index = np.array(coarse_index[outlier_tag == 1])
        refined_index = np.reshape(refined_index, newshape=(refined_index.size,))

        # Calculate F-score = 2.0/(1.0/precision + 1.0/recall)
        print refined_index
        print self.param.data_set[refined_index, :].shape
        f = self._pre_plot_3d(rescaled_data[refined_index, :])
        plt.show(f)


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
#
#
# def save_data(data_set, data_tag, file_name):
#     data = np.append(data_set, data_tag, axis=1)
#     np.savetxt(file_name, data)
#     return True


def main():
    """
    Main function of python.
    """
    # Config parameters
    # rare_index = 2
    # rare_size = 50
    # rare_sample_size = 2
    # major_index = 5
    # major_sample_size = 5
    # major_size = 500
    # dim = 3

    rare_index = 1
    rare_size = 10
    rare_sample_size = 2
    major_index = 2
    major_sample_size = 10
    major_size = 50
    dim = 4
    print "Loading synthetic data set..."
    source = np.loadtxt("sample_data_1.txt")
    iris = database.load_iris()
    min_max_scalar = preprocessing.MinMaxScaler()
    # data_set = min_max_scalar.fit_transform(source[:, 0:dim])
    # data_tag = source[:, -1]
    data_set = min_max_scalar.fit_transform(iris.data[50:, :])
    data_tag = iris.target[50: ]
    data_set, data_tag = pre_process(data_set, data_tag, 1, 10)
    print data_tag
    rare_param = RareParam(data_set=data_set, tag_set=data_tag, size=major_size + rare_size, dim=dim,
                           rare_index=rare_index, major_index=major_index, rare_sample_size=rare_sample_size,
                           major_sample_size=major_sample_size)

    rare_exp = RareCategoryExploration(rare_param)
    rare_exp.exploration(name="rare", c=0.5, stop_error=1e-6, max_iter=100, init_point=None)
    rare_exp.get_rescaled_data()
    # plt.show(f)
    # f.savefig('/home/haiqw/Dropbox/Latex/Mahalanobis/pic/euclidean.eps', format='eps', bbox_inches='tight',
    #           facecolor='w', edgecolor='w', transparent=True, dpi=1200)

    # Save data
    # np.savetxt("sample_data_1.txt", np.append(data, tag, axis=1))

    # Config searching parameter
    # rare_test = np.array([[ 4.50483295, -4.43064881], [4.60141853, -4.19811788]])
    # major_test = np.array([[-1.78122071, 7.74226893], [2.69124233, 3.29862953]])

if __name__ == "__main__":
    main()
