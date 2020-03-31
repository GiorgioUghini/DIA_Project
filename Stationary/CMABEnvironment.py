import numpy as np
import utils


# The real function to estimate
def fun(t, userType):
    return utils.getClickCurve(3, userType, t)  # Only the last phase, the one in high interest and with competitors


class CMABEnvironment():
    def __init__(self, budgets_list, sigma):
        self.budgets = budgets_list
        self.means = [fun(budgets_list[userType], userType) for userType in range(0, len(budgets_list))]
        self.sigma = sigma

    def round(self, pulled_arm, userType):
        mean = self.means[userType][pulled_arm]
        # We are supposing same variance among all userType:
        return np.random.normal(mean, self.sigma)
