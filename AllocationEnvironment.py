import numpy as np


def fun(x):
    return 100 * (1.0 - np.exp(-4*x + 3*x**3))


class AllocationEnvironment():
    def __init__(self, budgets, sigma):
        self.budgets = budgets
        self.means = fun(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])