import numpy as np


class Param:
    def __init__(self, n):
        self.mu_1 = np.array([-4.0, 5.0])
        self.mu_2 = np.array([4.0, -5.0])
        self.sigma_1 = np.array([[1, 0], [0, 1]])
        self.sigma_2 = np.array([[1, 0], [0, 1]])
        self.pi = np.array([0.7, 0.3])
        self.gamma = np.ones((n, 2)) / 2