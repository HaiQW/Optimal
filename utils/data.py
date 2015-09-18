#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


def gen_data_set(n_normals=100, n_moons=100, n_scurves=100, n_circles=100):
    """
    Generate synthetic data set with manifold rare category suite for RCD and
    RCE scenario.
    """
    if n_normals <= 0 and n_moons <= 0 and  n_scurves <= 0 and n_circles <= 0:
        return False

    # Generate all sub-manifold synthetic data sets.  First generate moon-shaped
    # data set, labels (0 or 1).
    moons_data, moons_labels = datasets.make_moons(n_samples=n_moons, noise=0.05)
    moons_data[:, 0] = moons_data[:, 0] * 0.5 + 5
    moons_data[:, 1] = moons_data[:, 1] * 0.5 + 5
    # Generate s_curve-shaped data set, labels (2).
    scurve_data, scurve_labels = datasets.samples_generator.make_s_curve(n_scurves, random_state=0)
    scurve_labels = np.ones(shape=(n_scurves)) * 2
    scurve_data[:, 0] = scurve_data[:, 0] * 0.5 + 5
    scurve_data[:, 2] = scurve_data[:, 2] * 0.5 - 5
    # Generate circle_data-shaped data set, labels (3 or 4).
    circle_data, circle_lables = datasets.make_circles(n_samples=n_circles)
    circle_lables = circle_lables + 3
    circle_data[:, 0] = circle_data[:, 0] * 0.5 - 5
    circle_data[:, 1] = circle_data[:, 1] * 0.5 - 5
    # Generate normal-shaped dataset, labels (5 or 6).
    normal_data, normal_labels = datasets.make_blobs(n_samples=n_normals,
                                                     n_features=2, centers=[0, 0])
    normal_labels = normal_labels + 5
    normal_data = normal_data  * 3

    # Combine sub-manifold datasets to form the final dataset.
    final_data = np.append(np.append(moons_data, scurve_data[:, [0, 2]], axis=0),
                           np.append(circle_data, normal_data, axis=0), axis=0)
    final_labels = np.append(np.append(moons_labels, scurve_labels, axis=0),
                             np.append(circle_lables, normal_labels, axis=0), axis=0)

    return final_data, final_labels


def plot_figure(dataset, labels, fig_name):
    """
    Plot the dataset to visualize it and save the figure to eps formated file.
    """
    fig = plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', linewidths=0, s=10, c=labels)
    plt.savefig('/home/haiqw/Dropbox/PycharmProjects/SyntheticData/%s' % fig_name,
                format='eps', bbox_inches='tight', dpi=1200)
    plt.show(fig)


def save_data(dataset, labels, file_name):
    """
    Write the synthetic dataset to file
    """
    np.savetxt(file_name, np.append(dataset, np.reshape(labels, newshape=(labels.size, 1)), axis=1))


def main():
    """
    Main function to test the data generator module.:
    """
    data_set, labels = gen_data_set(n_normals=1000, n_moons=100, n_scurves=100, n_circles=100)
    plot_figure(data_set, labels, 'name.eps')
    save_data(data_set, labels, './Synthetic_data.txt')

if __name__ == "__main__":
    main()