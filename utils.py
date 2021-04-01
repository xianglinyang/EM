import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_1, n_2, mu_1, mu_2, sigma_1, sigma_2):
    """
    construct a dataset using provided mu and sigma
    :param n_1: number of points in cluster 1
    :param n_2: number of points in cluster 2
    :param mu_1: mu in cluster 1
    :param mu_2: mu in cluster 2
    :param sigma_1: sigma in cluster 1
    :param sigma_2: sigma in cluster 2
    :return: data: the generated dataset [x, y, label(0 or 1)]
    """
    data_1 = np.random.multivariate_normal(mu_1, sigma_1, size=n_1)
    data_1 = np.concatenate((data_1, np.zeros((n_1, 1))), axis=1)

    data_2 = np.random.multivariate_normal(mu_2, sigma_2, size=n_2)
    data_2 = np.concatenate((data_2, np.ones((n_2, 1))), axis=1)

    data = np.concatenate((data_1, data_2), axis=0)
    return data


def check_converge(param, new_param):
    dist_mu = np.linalg.norm(param.mu_1 - new_param.mu_1) + np.linalg.norm(param.mu_2 - new_param.mu_2)
    dist_sigma = np.linalg.norm(param.sigma_1 - new_param.sigma_1) + np.linalg.norm(param.sigma_2 - new_param.sigma_2)
    return dist_mu+dist_sigma


def cal_prob(coordinate, mu, sigma):
    p = 1.0
    for i in range(2):
        p = p * normpdf(coordinate[i], mu[i], sigma[i, i])
    return p


def normpdf(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma))/(np.sqrt(2*np.pi*sigma))


def savefig(data, mu_1, mu_2, name):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="tab10", s=5)
    ax.scatter(mu_1[0], mu_1[1], c="red", s=20, marker="x")
    ax.scatter(mu_2[0], mu_2[1], c="blue", s=20, marker="x")
    ax.set_title("{}".format(name))
    fig.savefig("result\\{}.png".format(name))
