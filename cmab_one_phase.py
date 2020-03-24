import numpy as np
import matplotlib.pyplot as plt
from CMABEnvironment import *
from CMABOptimizer import *
from GPTS_Learner import *

step = 2
min_budgets = [0, 0, 0]
max_budgets = [34, 38, 36]
total_budget = 75
budgets_j = [np.arange(min_budgets[0], max_budgets[0] + 1, step), np.arange(min_budgets[1], max_budgets[1] + 1, step), np.arange(min_budgets[2], max_budgets[2] + 1, step)]      # +1 to max_budget because range does not include the right extreme of the interval by default
n_arms = [len(budgets_j[0]), len(budgets_j[1]), len(budgets_j[2])]
sigma = 100
T = 120
J = 3
n_experiments = 30
per_experiment_rewards_gpts = [[] for i in range(0, J)]

#   This script, configured with n_experiments = 30, T = 120, and ~19 arms for each userType
#   runs, in a i7-7000 machine with 16Gb of RAM in approximately 1 hour and a half

for e in range(0, n_experiments):
    opt = CMABOptimizer(max_budget=total_budget, campaign_number=J, step=step)
    envs = [CMABEnvironment(budgets=budgets_j[0], sigma=sigma, userType=0), CMABEnvironment(budgets=budgets_j[1], sigma=sigma, userType=1), CMABEnvironment(budgets=budgets_j[2], sigma=sigma, userType=2)]
    gpts_learners = [GPTS_Learner(n_arms=n_arms[0], arms=budgets_j[0]), GPTS_Learner(n_arms=n_arms[1], arms=budgets_j[1]), GPTS_Learner(n_arms=n_arms[2], arms=budgets_j[2])]

    for t in range(0, T):
        # Create matrix for the optimization process by sampling the GPTS
        colNum = int(np.floor_divide(total_budget, step) + 1)
        base_matrix = np.ones((J, colNum)) * np.NINF
        for j in range(0, J):
            sampled_values = gpts_learners[j].sample_values()
            bubblesNum = int(min_budgets[j] / step)
            indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(sampled_values))]
            np.put(base_matrix, indices_list, sampled_values)

        # Choose budget thanks to the samples in the matrix
        chosen_budget = opt.optimize(base_matrix)

        # Update model of the GPTS
        arms_chosen = []
        for j in range(0, J):
            chosen_arm = gpts_learners[j].convert_value_to_arm(chosen_budget[j])
            chosen_arm = int(chosen_arm[0])
            reward = envs[j].round(chosen_arm)
            gpts_learners[j].update(chosen_arm, reward)

    # Append rewards for statistical purposes
    for j in range(0, J):
        per_experiment_rewards_gpts[j].append(gpts_learners[j].collected_rewards)


for j in range(0, J):
    opt = np.max(envs[j].means)
    plt.figure(j)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - per_experiment_rewards_gpts[j], axis=0)), 'r')
    plt.legend(["UserType = " + str(j)])
    plt.show()