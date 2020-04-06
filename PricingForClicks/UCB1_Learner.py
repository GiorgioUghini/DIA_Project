from PricingForClicks.Learner import Learner
import math
import numpy as np


class UCB1_Learner(Learner):
    def __init__(self, arms):
        n_arms = len(arms)
        super().__init__(n_arms)
        self.results_per_arm = [(0, 0, arms[i]) for i in range(n_arms)]

    def pull_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t-1
        else:
            upper_bounds = list(map(self.calc_upper_bound, self.results_per_arm))
            return np.argmax(upper_bounds)

    def calc_upper_bound(self, params):
        (avg, n, price) = params
        return avg + math.sqrt(2 * math.log10(self.t) / n)

    def update(self, pulled_arm, successes):
        k = 1000000
        (avg, n, p) = self.results_per_arm[pulled_arm]
        reward = successes * p / k

        new_avg = (avg * n + reward) / (n+1)
        self.results_per_arm[pulled_arm] = (new_avg, n + 1, p)


