import matplotlib.pyplot as plt
from CMABOptimizer import *
from Stationary.GPTS_Learner import *
from PricingForClicks.TS_Learner import *
from BudgetAllocationAndPricing.Main_Environment import *
import numpy as np
import csv
from datetime import datetime


TIME_SPAN = 50
N_CLASSES = 3
N_EXPERIMENTS = 1

min_budgets = [10, 10, 10]
max_budgets = [54, 58, 52]
total_budget = 90
step = 2
sigma = 200

# budgets_j = [ [0, 2, 4, ..., 34] , [0, 2, 4, ..., 38], [0, 2, 4, ..., 36] ]
# +1 to max_budget because range does not include the right extreme of the interval by default
budgets_j = [np.arange(min_budgets[x], max_budgets[x] + 1, step) for x in range(0, N_CLASSES)]
bdg_n_arms = [len(budgets_j[x]) for x in range(0, N_CLASSES)]
per_experiment_revenues = []

#pr_n_arms = math.ceil((TIME_SPAN * np.log10(TIME_SPAN)) ** 0.25)  # the optimal number of arms
pr_n_arms = 6  # Non cambiare, è per fare venire il grafico zoommato nei primi giorni
pr_maxPrice = [400, 400, 400]
pr_minPrice = [100, 100, 100]

for e in range(0, N_EXPERIMENTS):
    opt = CMABOptimizer(max_budget=total_budget, campaigns_number=N_CLASSES, step=step)
    env = MainEnvironment(budgets_list=budgets_j, sigma=sigma,
                          pr_n_arms=[pr_n_arms for _ in range(0, N_CLASSES)],
                          pr_minPrice=pr_minPrice, pr_maxPrice=pr_maxPrice)
    gpts_learners = [ GPTS_Learner(n_arms=bdg_n_arms[v], arms=budgets_j[v])
                      for v in range(0, N_CLASSES) ]
    pr_ts_learners = [ TS_Learner(arms=env.pr_probabilities[v]) for v in range(0, N_CLASSES) ]
    experiment_revenues = []
    print("Esperimento: " + str(e + 1))

    best_arms = []
    best_prices = []
    best_values_per_click = []
    for c in range(N_CLASSES):
        prices = env.pr_probabilities[c][:, 0]
        best_arms.append(np.argmax(prices * utils.getDemandCurve(c, prices)))
        best_prices.append(prices[best_arms[c]])
        best_values_per_click.append(utils.getDemandCurve(c, best_prices[c]) * best_prices[c])

    for t in range(0, TIME_SPAN):
        pricing_pulled_arms = [pr_ts_learners[j].pull_arm() for j in range(N_CLASSES)]

        # Start optimization data structure creation
        secondStageRows = []
        # Create FIRST matrix by sampling pricing and budget
        for j in range(0, N_CLASSES):
            colNum = pr_n_arms
            price = pr_ts_learners[j].arms[pricing_pulled_arms[j]][0]
            conversion_rate = pr_ts_learners[j].get_conversion_rate(pricing_pulled_arms[j])
            clicks = gpts_learners[j].sample_values()
            budgets = gpts_learners[j].arms
            values_per_click = (price * clicks * conversion_rate - budgets) / clicks
            secondStageRows.append(values_per_click * clicks)
        # Create SECOND matrix for the optimization process
        colNum = int(np.floor_divide(total_budget, step) + 1)  # that is, int(np.floor_divide(total_budget, step) + 1)
        base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
        for j in range(0, N_CLASSES):
            sampled_values = secondStageRows[j]  # Row obtained in the first step maximizing elements
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
            pulled_arm = pricing_pulled_arms[j]
            successes = env.round_pricing(pulled_arm, clicks, j)    # Successful clicks with current budget
            failures = clicks - successes
            pr_ts_learners[j].update(pulled_arm, successes, failures)

            aggregated_revenue += successes * env.pr_probabilities[j][pulled_arm][0]    # For all classes

        # Append the revenue of this day to the array for this experiment
        experiment_revenues.append(aggregated_revenue)

        if t % 10 == 0:
            timestampStr = datetime.now().strftime("%H:%M:%S")
            print(timestampStr + " - Step %s of %s (%s exp)" % (
            (t / 10), TIME_SPAN / 10, e + 1))

    # Append the array of revenues of this experiment to the main one, for statistical purposes
    per_experiment_revenues.append(experiment_revenues)

print("Max revenue: %f" % np.max(per_experiment_revenues))


# Compute the REAL optimum allocation by solving the optimization problem with the real values
secondStageRows = []
# Create FIRST matrix by sampling pricing and budget
for j in range(0, N_CLASSES):
    demand_curve = demand(j, best_prices[j])
    colNum = pr_n_arms
    clicks = env.means[j]
    values_per_click = (best_prices[j] * clicks * demand_curve - budgets_j[j]) / clicks
    secondStageRows.append(values_per_click * clicks)  # remove dependency of price: sum along rows
# Create SECOND matrix for the optimization process
colNum = int(np.floor_divide(total_budget, step) + 1)  # that is, int(np.floor_divide(total_budget, step) + 1)
base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
for j in range(0, N_CLASSES):
    sampled_values = secondStageRows[j]  # Row obtained in the first step maximizing elements
    bubblesNum = int(min_budgets[j] / step)
    indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(sampled_values))]
    np.put(base_matrix, indices_list, sampled_values)
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
with open(timestamp + "-optimal-rew.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(cla_rew)
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
