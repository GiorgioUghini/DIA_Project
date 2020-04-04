from PricingForClicks.Learner import Learner
import math
import numpy as np


class UCB1_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.results_per_arm = [(0, 0) for _ in range(n_arms)]

    def pull_arm(self):
        super().pull_arm()
        if self.t <= self.n_arms:
            return self.t-1
        else:
            upper_bounds = list(map(self.calc_upper_bound, self.results_per_arm))
            return np.argmax(upper_bounds)

    def calc_upper_bound(self, params):
        (avg, n) = params
        return avg + math.sqrt(2 * math.log10(self.t) / n)

    def update(self, pulled_arm, successes, failures):
        normalized = successes / (successes + failures)
        (avg, n) = self.results_per_arm[pulled_arm]
        new_avg = (avg * n + normalized) / (n+1)
        self.results_per_arm[pulled_arm] = (new_avg, n + 1)


