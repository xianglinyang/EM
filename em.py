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
    update_gamma = np.zeros((len(data), 2))
    for i in range(len(data)):
        coordinate = data[i][:2]
        p_1 = cal_prob(coordinate, param.mu_1, param.sigma_1)
        p_2 = cal_prob(coordinate, param.mu_2, param.sigma_2)
        update_gamma[i][0] = p_1
        update_gamma[i][1] = p_2
        if p_1 > p_2:
            data[i][2] = 0
        else:
            data[i][2] = 1
    update_gamma[:, 0] = update_gamma[:, 0] * param.pi[0]
    update_gamma[:, 1] = update_gamma[:, 1] * param.pi[1]
    gamma_sum = update_gamma.sum(axis=1)
    gamma_sum = np.vstack((gamma_sum, gamma_sum)).T
    update_gamma = update_gamma / gamma_sum
    return data, update_gamma


def get_mean(data, gamma, k):
    mu = np.zeros(2)
    for i in range(len(data)):
        mu = mu + gamma[i][k]*data[i][:2]
    tau_k = gamma.sum(axis=0)[k]
    return mu / tau_k


def get_var(data, gamma, k, mu):
    var = np.zeros((2, 2))
    for i in range(len(data)):
        var = var + gamma[i][k] * np.outer(data[i][:2], data[i][:2])
    tau_k = gamma.sum(axis=0)[k]
    var = var / tau_k

    var = var - np.outer(mu, mu)
    return var


def maximization(data, update_gamma):
    update_param = Param.Param(len(data))

    update_param.pi[0] = update_gamma.sum(axis=0)[0] / len(data)
    update_param.pi[1] = 1 - update_param.pi[0]

    update_param.mu_1 = get_mean(data, update_gamma, 0)
    update_param.mu_2 = get_mean(data, update_gamma, 1)

    update_param.sigma_1 = get_var(data, update_gamma, 0, update_param.mu_1)
    update_param.sigma_2 = get_var(data, update_gamma, 1, update_param.mu_2)

    return update_param


def EM(data_exp, param):
    update = 10e5
    iteration = 0
    epsilon = 10e-3
    savefig(data_exp, param.mu_1, param.mu_2, "iteration{:d}".format(iteration))
    while update > epsilon:
        iteration = iteration + 1
        update_data, update_gamma = expectation(data_exp, param)
        update_param = maximization(update_data, update_gamma)
        update_param.gamma = np.copy(update_gamma)

        update = check_converge(param, update_param)
        print(
            "iteration:{:.2f}, update:{:.2f}, mu_1:({:.2f}, {:.2f}), mu_2:({:.2f}, {:.2f}), var_1:({:.2f}, {:.2f}), var_2:({:.2f}, {:.2f}), lambda:({:.2f}, {:.2f})".format(
                iteration, update,
                param.mu_1[0],
                param.mu_1[1],
                param.mu_2[0],
                param.mu_2[1],
                param.sigma_1[0, 0],
                param.sigma_1[1, 1],
                param.sigma_2[0, 0],
                param.sigma_2[1, 1],
                param.pi[0],
                param.pi[1]
            ))
        data_exp = update_data
        param = update_param

        savefig(data_exp, param.mu_1, param.mu_2, "iteration{:d}".format(iteration))
    return data_exp, param

