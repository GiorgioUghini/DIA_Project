import numpy as np
import utils


# The real function to estimate
def fun(userType, t):
    return utils.getClickCurve(3, userType, np.array(t))  # Only the last phase, the one in high interest and with competitors

# demand curve
def demand(userType, t):
    return utils.getDemandCurve(userType, np.array(t))  # demandCurve fn


class MainEnvironment():
    def __init__(self, budgets_list, sigma, pr_n_arms, pr_maxPrice, pr_minPrice):
        ### Budget Problem
        self.budgets = budgets_list
        self.N_USERS = len(budgets_list)
        self.means = [fun(userType, budgets_list[userType]) for userType in range(0, self.N_USERS)]
        self.sigma = sigma
        ### Pricing Problem
        self.pr_n_arms = pr_n_arms  # array of arrays
        pr_step = []
        for q in range(0, self.N_USERS):
            pr_step.append((pr_maxPrice[q] - pr_minPrice[q]) / (pr_n_arms[q] - 1))
        self.pr_probabilities = []
        for q in range(0, self.N_USERS):
            tmp_prb = []
            x = pr_minPrice[q]
            while x <= pr_maxPrice[q]:
                tmp_prb.append([x, demand(q, x)])
                x += pr_step[q]
            if len(self.pr_probabilities[q]) < pr_n_arms[q]:
                tmp_prb.append([pr_maxPrice[q], demand(q, pr_maxPrice)])
            self.pr_probabilities.append(np.array(tmp_prb))

    def round_budget(self, pulled_arm, userType):
        mean = self.means[userType][pulled_arm]
        # We are supposing same variance among all userType:
        return np.random.normal(mean, self.sigma)

    def round_pricing(self, pulled_arm, clicks):
        rewards = np.random.binomial(clicks, self.pr_probabilities[pulled_arm][1])  # TODO: Usertype?
        return rewards