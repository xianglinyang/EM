import numpy as np


class Param:
    def __init__(self):
        self.mu_1 = np.array([1, 1])
        self.mu_2 = np.array([5, 5])
        self.sigma_1 = np.array([[1, 0], [0, 1]])
        self.sigma_2 = np.array([[1, 0], [0, 1]])
        self.lambd = np.array([0.7, 0.3])