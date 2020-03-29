from Learner import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms)*10
        self.pulled_arms = []
        kernel = C(1e4, (1e-3, 1e4)) * RBF(1, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1, normalize_y=True, n_restarts_optimizer=25)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)   # Dati di ieri
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True) # Fai previsione così scelgo il budget da allocare oggi
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = self.sample_values()
        return np.argmax(sampled_values)

    def sample_values(self):
        return np.random.normal(self.means, self.sigmas)    # Mi ritorna un vettore [ 100, 200, 300, ... 1000 ]

    def convert_value_to_arm(self, value):
        return np.where(self.arms == value)

    '''
    Use this function if you want to plot the current status of the prediction
    just call the correct learner for a certain usertype and edit the correct function below.
    
    def plotFn(self):
        correct = 3500 * (1 - np.exp((-1*self.arms)/10))
        plt.figure(self.t)
        plt.plot(self.arms, correct, 'r:', 'real fn')
        plt.plot(self.pulled_arms, self.collected_rewards.ravel(), 'ro', 'observations')
        plt.plot(self.arms, self.means, 'b-', 'predicted')
        plt.fill(np.concatenate([self.arms, self.arms[::-1]]),
                 np.concatenate([self.means - 1.9600 * self.sigmas,
                                 (self.means + 1.9600 * self.sigmas)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.legend(loc='lower right')
        plt.show()
    '''
