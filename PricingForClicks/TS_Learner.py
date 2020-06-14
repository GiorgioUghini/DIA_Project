from PricingForClicks.Learner import *
from matplotlib import pyplot


class TS_Learner(Learner):
    def __init__(self, arms):
        n_arms = len(arms)
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.arms = arms
        self.arms_history = []

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * self.arms[:, 0])
        return idx

    def get_conversion_rate(self, arm):
        return np.random.beta(self.beta_parameters[arm, 0], self.beta_parameters[arm, 1])

    def sample_values(self):
        probabilities = np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        return probabilities

    def update(self, pulled_arm, successes, failures):
        self.beta_parameters[pulled_arm, 0] += successes
        self.beta_parameters[pulled_arm, 1] += failures
        self.update_observations(pulled_arm, successes)
        self.arms_history.append(pulled_arm)