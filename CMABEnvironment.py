import numpy as np


def fun(x, userType):
    if (userType == 0):
        return 10500 * (1 - np.exp((-1*x)/40))
    elif (userType == 2):
        return 12000 * (1 - np.exp((-1*x)/70)) + 1000 * np.log(x+1)
    else:
        return 3500 * (1 - np.exp((-1*x)/10))


class CMABEnvironment():
    def __init__(self, budgets, sigma, userType):
        self.budgets = budgets
        self.means = fun(budgets, userType)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])