from PricingForClicks.Learner import *
from matplotlib import pyplot


class TS_Learner(Learner):
    def __init__(self, arms):
        n_arms = len(arms)
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.arms = arms

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, successes):
        price = self.arms[pulled_arm, 0]
        k = 1000000
        reward = successes * price / k
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1-reward
