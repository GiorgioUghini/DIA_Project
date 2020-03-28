import matplotlib.pyplot as plt
from CMABOptimizer import *
from Stationary.GPTS_Learner import *

from Non_Stationary.NSCMABEnvironment import *
from Non_Stationary.GPSWTS_Learner import *


# Assumption: Static min/max budget allocation for all three phases
step = 2
min_budgets = [0, 0, 0]
max_budgets = [34, 38, 36]
total_budget = 75
budgets_j = [np.arange(min_budgets[0], max_budgets[0] + 1, step), np.arange(min_budgets[1], max_budgets[1] + 1, step), np.arange(min_budgets[2], max_budgets[2] + 1, step)]      # +1 to max_budget because range does not include the right extreme of the interval by default
n_arms = [len(budgets_j[0]), len(budgets_j[1]), len(budgets_j[2])]
sigma = 100
T = 120  # T should be a multiple of env.N_PHASES or not all phases wil have same lenght
J = 3
n_experiments = 4
per_experiment_rewards_gpts = [[] for i in range(0, J)]
per_experiment_rewards_gpswts = [[] for j in range(0, J)]
window_size = int(np.sqrt(T))


for e in range(0, n_experiments):
    opt = CMABOptimizer(max_budget=total_budget, campaign_number=J, step=step)

    env = NSCMABEnvironment(budgets_list=budgets_j, sigma=sigma, horizon=T)

    gpts_learners = [GPTS_Learner(n_arms=n_arms[0], arms=budgets_j[0]), GPTS_Learner(n_arms=n_arms[1], arms=budgets_j[1]), GPTS_Learner(n_arms=n_arms[2], arms=budgets_j[2])]

    gpsw_learners = [GPSWTS_Learner(n_arms=n_arms[0], arms=budgets_j[0], window_size=window_size), GPSWTS_Learner(n_arms=n_arms[1], arms=budgets_j[1], window_size=window_size), GPSWTS_Learner(n_arms=n_arms[2], arms=budgets_j[2], window_size=window_size)]


    for t in range(0, T):
        # TS: Create matrix for the optimization process by sampling the GPTS
        ts_colNum = int(np.floor_divide(total_budget, step) + 1)
        ts_base_matrix = np.ones((J, ts_colNum)) * np.NINF
        for j in range(0, J):
            ts_sampled_values = gpts_learners[j].sample_values()
            ts_bubblesNum = int(min_budgets[j] / step)
            ts_indices_list = [i for i in range(ts_bubblesNum + ts_colNum * j, ts_bubblesNum + ts_colNum * j + len(ts_sampled_values))]
            np.put(ts_base_matrix, ts_indices_list, ts_sampled_values)


        # SWTS: Create matrix for the optimization process by sampling the GPTS
        sw_colNum = int(np.floor_divide(total_budget, step) + 1)
        sw_base_matrix = np.ones((J, sw_colNum)) * np.NINF
        for j in range(0, J):
            sw_sampled_values = gpsw_learners[j].sample_values()
            sw_bubblesNum = int(min_budgets[j] / step)
            sw_indices_list = [i for i in range(sw_bubblesNum + sw_colNum * j, sw_bubblesNum + sw_colNum * j + len(sw_sampled_values))]
            np.put(sw_base_matrix, sw_indices_list, sw_sampled_values)

        # Choose budget thanks to the samples in the matrix
        ts_chosen_budget = opt.optimize(ts_base_matrix)
        sw_chosen_budget = opt.optimize(sw_base_matrix)

        # Update model of the GPTS
        ts_arms_chosen = []
        for j in range(0, J):
            ts_chosen_arm = gpts_learners[j].convert_value_to_arm(ts_chosen_budget[j])
            ts_chosen_arm = int(ts_chosen_arm[0])
            ts_reward = env.round(ts_chosen_arm, j)
            gpts_learners[j].update(ts_chosen_arm, ts_reward)

        # Update model of the GPSWTS
        sw_arms_chosen = []
        for j in range(0, J):
            sw_chosen_arm = gpsw_learners[j].convert_value_to_arm(sw_chosen_budget[j])
            sw_chosen_arm = int(sw_chosen_arm[0])
            sw_reward = env.round(sw_chosen_arm, j)
            gpsw_learners[j].update(sw_chosen_arm, sw_reward)

        env.ahead()

    # Append rewards for statistical purposes
    for j in range(0, J):
        per_experiment_rewards_gpts[j].append(gpts_learners[j].collected_rewards)
        per_experiment_rewards_gpswts[j].append(gpsw_learners[j].collected_rewards)

opt_per_phase = []
for p in range(0, env.N_PHASES):
    # Compute the REAL optimum allocation by solving the optimization problem with the real values
    colNum = int(np.floor_divide(total_budget, step) + 1)
    base_matrix = np.ones((J, colNum)) * np.NINF
    for j in range(0, J):
        real_values = env.means[j]
        bubblesNum = int(min_budgets[j] / step)
        indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(real_values))]
        np.put(base_matrix, indices_list, real_values)
    chosen_budget = opt.optimize(base_matrix)
    chosen_arms = [gpts_learners[j].convert_value_to_arm(chosen_budget[j])[0] for j in range(0, J)]
    optimized_alloc = [env.means[j][chosen_arms[j]] for j in range(0, J)]
    opt_per_phase.append(np.sum(optimized_alloc))

# Print aggregated results
aggr_rewards = np.sum(per_experiment_rewards_gpts, axis=0)
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")

optimal_vector = []
for t in range(0, T):
    phase_size = T / env.N_PHASES
    current_phase = int(t / phase_size)
    optimal_vector.append(opt_per_phase[current_phase])
a = optimal_vector - aggr_rewards

c = np.cumsum(np.mean(a, axis=0))
plt.plot(c, 'r')
plt.legend(["UserType = "])
plt.show()