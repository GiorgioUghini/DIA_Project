import matplotlib.pyplot as plt
from CMABOptimizer import *
from Stationary.GPTS_Learner import *
from PricingForClicks.TS_Learner import *
from BudgetAllocationAndPricing.Main_Environment import *
import numpy as np
import math
import csv
from datetime import datetime
from scipy import optimize


TIME_SPAN = 180
N_CLASSES = 3
N_EXPERIMENTS = 50

min_budgets = [10, 10, 10]
max_budgets = [54, 58, 52]
total_budget = 90
step = 2
sigma = 100

# budgets_j = [ [0, 2, 4, ..., 34] , [0, 2, 4, ..., 38], [0, 2, 4, ..., 36] ]
# +1 to max_budget because range does not include the right extreme of the interval by default
budgets_j = [np.arange(min_budgets[x], max_budgets[x] + 1, step) for x in range(0, N_CLASSES)]
bdg_n_arms = [len(budgets_j[x]) for x in range(0, N_CLASSES)]
per_experiment_revenues = []

pr_n_arms = math.ceil((TIME_SPAN * np.log10(TIME_SPAN)) ** 0.25)  # the optimal number of arms
best_prices = [optimize.fmin(lambda x: -utils.getDemandCurve(j, x) * x, TIME_SPAN / 2)[0]
               for j in range(0, N_CLASSES)]
best_values_per_click = [utils.getDemandCurve(j, best_prices[j]) * best_prices[j]
                         for j in range(0, N_CLASSES)]


for e in range(0, N_EXPERIMENTS):
    opt = CMABOptimizer(max_budget=total_budget, campaign_number=N_CLASSES, step=step)
    env = MainEnvironment(budgets_list=budgets_j, sigma=sigma,
                          pr_n_arms=[pr_n_arms for _ in range(0, N_CLASSES)],
                          pr_minPrice=[100, 100, 100], pr_maxPrice=[400, 400, 400])
    gpts_learners = [ GPTS_Learner(n_arms=bdg_n_arms[v], arms=budgets_j[v])
                      for v in range(0, N_CLASSES) ]
    pr_ts_learners = [ TS_Learner(arms=env.pr_probabilities[v]) for v in range(0, N_CLASSES) ]
    experiment_revenues = []
    print("Esperimento: " + str(e))

    for t in range(0, TIME_SPAN):
        # Create matrix for the optimization process by sampling the GPTS
        colNum = int(np.floor_divide(total_budget, step) + 1)
        base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
        for j in range(0, N_CLASSES):
            sampled_values = gpts_learners[j].sample_values()
            bubblesNum = int(min_budgets[j] / step)
            indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(sampled_values))]
            np.put(base_matrix, indices_list, sampled_values)

        # Choose budget thanks to the samples in the matrix
        chosen_budget = opt.optimize(base_matrix)

        aggregated_revenue = 0
        for j in range(0, N_CLASSES):
            # Update model of the GPTS
            chosen_arm = gpts_learners[j].convert_value_to_arm(chosen_budget[j])
            chosen_arm = int(chosen_arm[0])
            clicks = np.round(env.round_budget(chosen_arm, j))
            gpts_learners[j].update(chosen_arm, clicks)

            # Pricing: problem can be decomposed. The optimal solution is the union
            # of the three optimal sub-solution as the seller can set a different price
            pulled_arm = pr_ts_learners[j].pull_arm()
            successes = env.round_pricing(pulled_arm, clicks, j)    # Successful clicks with current budget
            failures = clicks - successes
            pr_ts_learners[j].update(pulled_arm, successes, failures)
            aggregated_revenue += successes * env.pr_probabilities[j][pulled_arm][0]    # For all classes

        # Append the revenue of this day to the array for this experiment
        experiment_revenues.append(aggregated_revenue)

    # Append the array of revenues of this experiment to the main one, for statistical purposes
    per_experiment_revenues.append(experiment_revenues)


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
optimized_revenue = [optimized_alloc[j] * best_values_per_click[j] for j in range(0, N_CLASSES)]

# Storing results
timestamp = str(datetime.timestamp(datetime.now()))
exp_rew = np.mean(per_experiment_revenues, axis=0)
with open(timestamp + "-rew.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(exp_rew)
writeFile.close()
cla_rew = np.sum(optimized_revenue) * np.ones(len(exp_rew))
with open(timestamp + "-rew.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(exp_rew)
writeFile.close()

# Print aggregated results
aggr_optimal_revenue = np.sum(optimized_revenue)
aggr_rewards = np.mean(per_experiment_revenues, axis=0)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("time")
a = aggr_optimal_revenue - aggr_rewards
c = np.cumsum(a)
plt.plot(c, 'r')
plt.legend(["Budget + Pricing"])
plt.show()

plt.figure(1)
plt.ylabel("Revenue")
plt.xlabel("time")
a = aggr_rewards
b = aggr_optimal_revenue * np.ones(len(aggr_rewards))
plt.plot(a, 'b')
plt.plot(b, 'k--')
plt.legend(["Algorithm", "Optimal"])
plt.show()

