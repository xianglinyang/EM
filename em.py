import numpy as np
from utils import *
import Param


def expectation(data_exp, param):
    """
    E step
    :param data: [x, y, label]
    :param param: mu sigma lambda
    :return: data, the dataset with update label
    """
    data = np.copy(data_exp)
    for i in range(len(data)):
        coordinate = data[i][:2]
        p_1 = cal_prob(coordinate, param.mu_1, param.sigma_1, param.lambd[0])
        p_2 = cal_prob(coordinate, param.mu_2, param.sigma_2, param.lambd[1])
        if p_1 > p_2:
            data[i][2] = 0
        else:
            data[i][2] = 1
    return data


def maximization(data, param):
    cluster_1 = data[np.where(data[:, 2] == 0)]
    cluster_2 = data[np.where(data[:, 2] == 1)]

    update_param = Param.Param()

    update_param.lambd[0] = len(cluster_1) / float(len(data))
    update_param.lambd[1] = 1 - update_param.lambd[0]

    update_param.mu_1[0] = np.mean(cluster_1[:, 0])
    update_param.mu_1[1] = np.mean(cluster_1[:, 1])
    update_param.mu_2[0] = np.mean(cluster_2[:, 0])
    update_param.mu_2[1] = np.mean(cluster_2[:, 1])

    update_param.sigma_1[0, 0] = np.var(cluster_1[:, 0])
    update_param.sigma_1[1, 1] = np.var(cluster_1[:, 1])
    update_param.sigma_2[0, 0] = np.var(cluster_2[:, 0])
    update_param.sigma_2[1, 1] = np.var(cluster_2[:, 1])

    return update_param


def EM(data_exp, param):
    update = 10e5
    iteration = 0
    epsilon = 10e-3
    savefig(data_exp, param.mu_1, param.mu_2, "iteration{:d}".format(iteration))
    while update > epsilon:
        iteration = iteration + 1
        update_data = expectation(data_exp, param)
        update_param = maximization(update_data, param)

        update = check_converge(param, update_param)
        print(
            "iteration:{:.2f}, update:{:.2f}, mu_1:({:.2f}, {:.2f}), mu_2:({:.2f}, {:.2f}), var_1:({:.2f}, {:.2f}), var_2:({:.2f}, {:.2f}), ".format(
                iteration, update,
                param.mu_1[0],
                param.mu_1[1],
                param.mu_2[0],
                param.mu_2[1],
                param.sigma_1[0, 0],
                param.sigma_1[1, 1],
                param.sigma_2[0, 0],
                param.sigma_2[1, 1],

            ))
        data_exp = update_data
        param = update_param

        savefig(data_exp, param.mu_1, param.mu_2, "iteration{:d}".format(iteration))
    return data_exp, param

