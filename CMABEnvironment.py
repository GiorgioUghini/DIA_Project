import numpy as np


# The real function to estimate
def fun(x, userType):
    if (userType == 0):
        return 10500 * (1 - np.exp((-1*x)/40))
    elif (userType == 1):
        return 12000 * (1 - np.exp((-1*x)/70)) + 1000 * np.log(x+1)
    else:
        return 3500 * (1 - np.exp((-1*x)/10))


class CMABEnvironment():
    def __init__(self, budgets_list, sigma):
        self.budgets = budgets_list
        self.means = [fun(budgets_list[userType], userType) for userType in range(0, len(budgets_list))]
        self.sigma = sigma

    def round(self, pulled_arm, userType):
        means = self.means[userType][pulled_arm]
        # We are supposing same variance among all userType:
        return np.random.normal(means, np.ones(len(means)) * self.sigma)
