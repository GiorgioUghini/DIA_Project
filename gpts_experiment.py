import numpy as np
import matplotlib.pyplot as plt
from AllocationEnvironment import *
from GPTS_Learner import *


n_arms = 21
min_budget = 0.0
max_budget = 1.0
budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 10
T = 30
n_experiments = 80
gpts_rewards_per_experiment = []


for e in range(0, n_experiments):
    env = AllocationEnvironment(budgets=budgets, sigma=sigma)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=budgets)

    for t in range(0, T):
        # GP Thompson Sampling Learner
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)

    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)


opt = np.max(env.means)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - gpts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["GPTS"])
plt.show()
