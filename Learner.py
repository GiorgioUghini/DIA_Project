import numpy as np

class Learner:
    def __init__(self,n_arms):
        self.n_arms = n_arms
        self.t = 0

    def pull_arm(self):
        self.t += 1
        pass
