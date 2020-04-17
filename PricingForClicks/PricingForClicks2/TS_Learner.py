import numpy as np


class TS_Learner():
    def __init__(self, arms):
        self.n_arms = len(arms)
        self.beta_parameters = np.ones((self.n_arms, 2))
        self.arms = arms
        self.arms_history = []

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * self.arms[:, 0])
        return idx

    def update(self, pulled_arm, result):
        self.beta_parameters[pulled_arm, 0] += result
        self.beta_parameters[pulled_arm, 1] += (1-result)
