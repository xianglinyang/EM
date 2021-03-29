import numpy as np
from utils import *
import Param
import em
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # define hyperparameters
    num_cluster_1 = 200
    num_cluster_2 = 300
    mu_1 = np.array([3, 3])
    mu_2 = np.array([-3, -3])
    sigma_1 = np.array([[2, 0], [0, 4.5]])
    sigma_2 = np.array([[1, 0], [0, 8]])
    data = generate_data(num_cluster_1, num_cluster_2, mu_1, mu_2, sigma_1, sigma_2)
    savefig(data, mu_1, mu_2, "ground_truth")

    # set a dataset with randomly assigned labels
    data_exp = np.copy(data)
    data_exp[:, 2] = np.random.randint(2, size=num_cluster_1+num_cluster_2)

    # start with a randomly guess
    param = Param.Param()

    data, param = em.EM(data_exp, param)


