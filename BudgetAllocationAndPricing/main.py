import matplotlib.pyplot as plt
from CMABOptimizer import *
from Stationary.GPTS_Learner import *
from PricingForClicks.TS_Learner import *
from BudgetAllocationAndPricing.Main_Environment import *
import math


T = 5
N_CLASSES = 3
n_experiments = 1

step = 2
min_budgets = [10, 10, 10]
max_budgets = [54, 58, 52]
sigma = 100
total_budget = 90
# budgets_j = [ [0, 2, 4, ..., 34] , [0, 2, 4, ..., 38], [0, 2, 4, ..., 36] ]
budgets_j = [np.arange(min_budgets[0], max_budgets[0] + 1, step), np.arange(min_budgets[1], max_budgets[1] + 1, step), np.arange(min_budgets[2], max_budgets[2] + 1, step)]      # +1 to max_budget because range does not include the right extreme of the interval by default
bdg_n_arms = [len(budgets_j[0]), len(budgets_j[1]), len(budgets_j[2])]
per_experiment_rewards_gpts = [[] for i in range(0, N_CLASSES)]

pr_estimated_initial_clicks = [2000, 2000, 2000]
pr_n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms


for e in range(0, n_experiments):
    opt = CMABOptimizer(max_budget=total_budget, campaign_number=N_CLASSES, step=step)
    env = MainEnvironment(budgets_list=budgets_j, sigma=sigma, pr_n_arms=[pr_n_arms, pr_n_arms, pr_n_arms], pr_minPrice=[100,100,100], pr_maxPrice=[400,400,400])
    gpts_learners = [ GPTS_Learner(n_arms=bdg_n_arms[v], arms=budgets_j[v]) for v in range(0, N_CLASSES) ]
    pr_ts_learners = [ TS_Learner(arms=env.pr_probabilities[v]) for v in range(0, N_CLASSES) ]
    clicks_yesterday = [ 2000, 2000, 2000 ]

    for t in range(0, T):
        # Create matrix for the optimization process by sampling the GPTS
        colNum = int(np.floor_divide(total_budget, step) + 1)
        base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
        for j in range(0, N_CLASSES):
            ####
            ######  Pricing based on yesterday's clicks
            ####
            pulled_arm = pr_ts_learners[j].pull_arm()
            successes = env.round_pricing(pulled_arm, clicks_yesterday[j])
            actual_value = successes * env.pr_probabilities[pulled_arm][0]
            ####

            sampled_values = actual_value * gpts_learners[j].sample_values()
            bubblesNum = int(min_budgets[j] / step)
            indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(sampled_values))]
            np.put(base_matrix, indices_list, sampled_values)

        # Choose budget thanks to the samples in the matrix
        chosen_budget = opt.optimize(base_matrix)

        # Update model of the GPTS
        arms_chosen = []
        for j in range(0, N_CLASSES):
            chosen_arm = gpts_learners[j].convert_value_to_arm(chosen_budget[j])
            chosen_arm = int(chosen_arm[0])
            clicks_yesterday[j] = env.round_budget(chosen_arm, j)
            gpts_learners[j].update(chosen_arm, clicks_yesterday[j])

    # Append rewards for statistical purposes
    for j in range(0, N_CLASSES):
        per_experiment_rewards_gpts[j].append(gpts_learners[j].collected_rewards)


# Compute the REAL optimum allocation by solving the optimization problem with the real values
colNum = int(np.floor_divide(total_budget, step) + 1)
base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
for j in range(0, N_CLASSES):
    real_values = env.means[j]
    bubblesNum = int(min_budgets[j] / step)
    indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(real_values))]
    np.put(base_matrix, indices_list, real_values)
chosen_budget = opt.optimize(base_matrix)
chosen_arms = [gpts_learners[j].convert_value_to_arm(chosen_budget[j])[0] for j in range(0, N_CLASSES)]
optimized_alloc = [env.means[j][chosen_arms[j]] for j in range(0, N_CLASSES)]

# Print aggregated results
aggr_optimal_value = np.sum(optimized_alloc)
aggr_rewards = np.sum(per_experiment_rewards_gpts, axis=0)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
a = aggr_optimal_value - aggr_rewards
b = np.mean(a, axis=0)
c = np.cumsum(b)
plt.plot(c, 'r')
plt.legend(["GPTS Learner"])
plt.show()

