import numpy as np


class Environment:
    def __init__(self, n_arms, demandCurve, minPrice, maxPrice):
        self.probabilities = []
        self.n_arms = n_arms
        self.demandCurve = demandCurve
        step = (maxPrice - minPrice) / (n_arms-1)
        x = minPrice
        while x <= maxPrice:
            self.probabilities.append(demandCurve(x))
            x += step

        if len(self.probabilities) < n_arms:
            self.probabilities.append(demandCurve(maxPrice))

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward


