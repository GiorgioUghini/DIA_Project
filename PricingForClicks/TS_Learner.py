from Learner import *
from matplotlib import pyplot

class TS_Learner(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, successes, failures):
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + successes
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + failures
