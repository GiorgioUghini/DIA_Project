import math
import numpy as np


class UCB1_Learner():
    def __init__(self, arms):
        self.n_arms = len(arms)
        self.results_per_arm = [(0, 0, arms[i, 0]) for i in range(self.n_arms)]
        self.t = 0

    def pull_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t-1
        else:
            upper_bounds = list(map(self.calc_upper_bound, self.results_per_arm))
            return np.argmax(upper_bounds)

    def calc_upper_bound(self, params):
        (avg, n, price) = params
        return price * (avg + math.sqrt(2 * math.log10(self.t) / n))

    def update(self, pulled_arm, reward):
        (avg, n, p) = self.results_per_arm[pulled_arm]
        new_avg = (avg * n + reward) / (n+1)
        self.results_per_arm[pulled_arm] = (new_avg, n + 1, p)