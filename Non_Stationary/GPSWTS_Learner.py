from Stationary.GPTS_Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPSWTS_Learner(GPTS_Learner):
    def __init__(self, n_arms, arms, window_size):
        super().__init__(n_arms, arms)
        self.window_size = window_size

    def update_model(self):
        if (len(self.pulled_arms) <= self.window_size):
            x = np.atleast_2d(self.pulled_arms).T
            y = self.collected_rewards
        else:
            x = np.atleast_2d(self.pulled_arms[-self.window_size:]).T
            y = self.collected_rewards[-self.window_size:]

        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
