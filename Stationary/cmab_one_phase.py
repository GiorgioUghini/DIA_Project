import matplotlib.pyplot as plt
from Stationary.CMABEnvironment import *
from CMABOptimizer import *
from Stationary.GPTS_Learner import *


N_CLASSES = 3
step = 2
min_budgets = [10, 10, 10]
max_budgets = [54, 58, 52]
total_budget = 90
# budgets_j = [ [0, 2, 4, ..., 34] , [0, 2, 4, ..., 38], [0, 2, 4, ..., 36] ]
budgets_j = [ np.arange(min_budgets[v], max_budgets[v] + 1, step) for v in range(0, N_CLASSES) ]      # +1 to max_budget because range does not include the right extreme of the interval by default
n_arms = [ len(budgets_j[v]) for v in range(0, N_CLASSES) ]
sigma = 200
T = 41
n_experiments = 5
per_experiment_rewards_gpts = [[] for i in range(0, N_CLASSES)]

for e in range(0, n_experiments):
    opt = CMABOptimizer(max_budget=total_budget, campaign_number=N_CLASSES, step=step)
    env = CMABEnvironment(budgets_list=budgets_j, sigma=sigma)
    gpts_learners = [ GPTS_Learner(n_arms=n_arms[v], arms=budgets_j[v]) for v in range(0, N_CLASSES) ]

    for t in range(0, T):
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

        # Update model of the GPTS
        arms_chosen = []
        for j in range(0, N_CLASSES):
            chosen_arm = gpts_learners[j].convert_value_to_arm(chosen_budget[j])
            chosen_arm = int(chosen_arm[0])
            reward = env.round(chosen_arm, j)
            gpts_learners[j].update(chosen_arm, reward)

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

