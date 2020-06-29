import sys
sys.path.insert(0, "..")
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
N_EXPERIMENTS = 100

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

timestamp = str(datetime.now())
writeFile = open("data-%s.csv" % timestamp, "w")
writer = csv.writer(writeFile)
writer.writerow(["e", "t", "price", "sampled prices", "conv rates", "expected revenues", "budgets", "real clicks", "real buys", "real_revenue"])

for e in range(0, N_EXPERIMENTS):
    # environment
    env = MainEnvironment(budgets_list=budgets_j, sigma=sigma,
                          pr_n_arms=[pr_n_arms for _ in range(0, N_CLASSES)],
                          pr_minPrice=pr_minPrice, pr_maxPrice=pr_maxPrice)
    # learners and optimizer
    opt = CMABOptimizer(max_budget=total_budget, campaigns_number=N_CLASSES, step=step)
    gpts_learners = [ GPTS_Learner(n_arms=bdg_n_arms[v], arms=budgets_j[v])
                      for v in range(0, N_CLASSES) ]
    pr_ts_learners = [TS_Learner(arms=env.pr_probabilities[v]) for v in range(0, N_CLASSES)]
    experiment_revenues = []
    print("Esperimento: " + str(e + 1))

    for t in range(0, TIME_SPAN):
        # pull a price from each TS learner
        sampled_prices_arms = [pr_ts_learners[j].pull_arm() for j in range(N_CLASSES)]
        sampled_prices = [pr_ts_learners[j].arms[sampled_prices_arms[j]][0] for j in range(N_CLASSES)]
        # and get the corresponding conversion rate
        conversion_rates = [pr_ts_learners[j].get_conversion_rate(sampled_prices_arms[j]) for j in range(0, N_CLASSES)]
        # here we will store the expected revenue for each price
        expected_revenues = np.zeros(3)
        chosen_budgets = []
        # for each price pulled
        for j in range(0, N_CLASSES):
            # get the expected number of clicks from all GP learners, for all budgets
            clicks = [gpts_learners[i].sample_values() for i in range(0, N_CLASSES)]
            #estimate the values per click, one for each budget of each class
            values_per_click = []
            for i in range(0, N_CLASSES):
                p = sampled_prices[j]
                c = clicks[i]
                conv_rate = conversion_rates[j]
                budgets = gpts_learners[i].arms * 1000
                values_per_click.append((p * c * conv_rate - budgets) / c)

            # now we can solve the knapsack problem to find best allocation and expected revenue
            second_stage_rows = []
            for i in range(0, N_CLASSES):
                second_stage_rows.append(values_per_click[i] * clicks[i])

            colNum = int(np.floor_divide(total_budget, step) + 1)  # that is, int(np.floor_divide(total_budget, step) + 1)
            base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
            for i in range(0, N_CLASSES):
                sampled_values = second_stage_rows[i]  # Row obtained in the first step maximizing elements
                bubblesNum = int(min_budgets[i] / step)
                indices_list = [k for k in
                                range(bubblesNum + colNum * i, bubblesNum + colNum * i + len(sampled_values))]
                np.put(base_matrix, indices_list, sampled_values)

            # Choose budget thanks to the samples in the matrix
            chosen_budgets.append(opt.optimize(base_matrix))
            # estimate the revenue with this budget allocation and this price
            expected_revenues[j] = opt.best_revenue

        # after the estimation is done with all prices, pick the price that corresponds to the maximum expected value
        best_TS_learner = int(np.argmax(expected_revenues))
        best_price_arm = sampled_prices_arms[best_TS_learner]
        best_price = pr_ts_learners[best_TS_learner].arms[best_price_arm][0]
        # and pick the arms that correspond to the best budget chosen
        best_budget_arms = [gpts_learners[i].convert_value_to_arm(chosen_budgets[best_TS_learner][i])[0][0] for i in range(N_CLASSES)]

        # test with environment
        daily_revenue = 0
        real_clicks_list = []
        real_buys_list = []
        for i in range(0, N_CLASSES):
            real_clicks = int(env.round_budget(best_budget_arms[i], i))
            real_clicks_list.append(real_clicks)
            real_buys = env.round_pricing(best_price_arm, real_clicks, i)
            real_buys_list.append(real_buys)
            daily_revenue += real_buys * best_price - chosen_budgets[best_TS_learner][i]*1000
            # and update learners
            gpts_learners[i].update(best_budget_arms[i], real_clicks)
            pr_ts_learners[i].update(best_price_arm, real_buys, real_clicks - real_buys)


        # Append the revenue of this day to the array for this experiment
        experiment_revenues.append(daily_revenue)
        writer.writerow([e, t, best_price, sampled_prices, conversion_rates[best_TS_learner], expected_revenues, chosen_budgets[best_TS_learner], real_clicks_list, real_buys_list, daily_revenue])

        if t % 10 == 0:
            timestampStr = datetime.now().strftime("%H:%M:%S")
            print(timestampStr + " - Step %s of %s (%s exp)" % (
            (t / 10), TIME_SPAN / 10, e + 1))

    # Append the array of revenues of this experiment to the main one, for statistical purposes
    per_experiment_revenues.append(experiment_revenues)

writeFile.close()

# Compute the REAL optimum allocation by solving the optimization problem with the real values for all prices and then get the best reward
rewards_per_price = []
prices = env.pr_probabilities[0][:, 0]
for price in prices:
    second_stage_rows = []
    for userType in range(0, N_CLASSES):
        clicks = env.means[userType]
        demandCurve = demand(userType, price)
        values_per_click = (price * clicks * demandCurve - budgets_j[userType]*1000) / clicks
        second_stage_rows.append(values_per_click * clicks)

    colNum = int(np.floor_divide(total_budget, step) + 1)  # that is, int(np.floor_divide(total_budget, step) + 1)
    base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
    for j in range(0, N_CLASSES):
        sampled_values = second_stage_rows[j]  # Row obtained in the first step maximizing elements
        bubblesNum = int(min_budgets[j] / step)
        indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(sampled_values))]
        np.put(base_matrix, indices_list, sampled_values)
    chosen_budget = opt.optimize(base_matrix)
    rewards_per_price.append(opt.best_revenue)

optimized_revenue_index = np.argmax(rewards_per_price)
optimized_revenue = rewards_per_price[optimized_revenue_index]
best_price = prices[optimized_revenue_index]
print("Best price: " + str(best_price))
print("Optimized revenue: " + str(optimized_revenue))

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

timestamp = str(datetime.now())
# Print aggregated results
aggr_optimal_revenue = np.sum(optimized_revenue)
aggr_rewards = np.mean(per_experiment_revenues, axis=0)
plt.figure(0)
plt.ylabel("Regret [€]")
plt.xlabel("time [days]")
a = aggr_optimal_revenue - aggr_rewards
c = np.cumsum(a)
plt.plot(c, 'r')
if N_EXPERIMENTS > 10:
    plt.savefig("regret-%s.png" % timestamp)
plt.show()

plt.figure(1)
plt.ylabel("Reward [€]")
plt.xlabel("time [days]")
a = aggr_rewards
b = aggr_optimal_revenue * np.ones(len(aggr_rewards))
plt.plot(a, 'b')
plt.plot(b, 'k--')
plt.legend(["Algorithm", "Optimal"])
if N_EXPERIMENTS > 10:
    plt.savefig("reward-%s.png" % timestamp)
plt.show()
