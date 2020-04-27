import matplotlib.pyplot as plt
from CMABOptimizer import *
from Stationary.GPTS_Learner import *
from datetime import datetime
from Non_Stationary.NSCMABEnvironment import *
from Non_Stationary.GPSWTS_Learner import *
import csv


# Assumption: Static min/max budget allocation for all three phases
N_CLASSES = 3
N_EXPERIMENTS = 10
TIME_SPAN = 120  # TIME_SPAN should be a multiple of env.N_PHASES or not all phases wil have same length
min_budgets = [10, 10, 10]
max_budgets = [70, 80, 60]
step = 2
total_budget = 110
WINDOW_SIZE = int(4 * np.power(TIME_SPAN * np.log(TIME_SPAN), 0.25))  #  4 * quadroot(T log(T))
sigma = 200

budgets_j = [np.arange(min_budgets[v], max_budgets[v] + 1, step) for v in range(0, N_CLASSES) ]      # +1 to max_budget because range does not include the right extreme of the interval by default
n_arms = [len(budgets_j[v]) for v in range(0, N_CLASSES) ]
per_experiment_rewards_gpts = [[] for i in range(0, N_CLASSES)]
per_experiment_rewards_gpswts = [[] for j in range(0, N_CLASSES)]
colNum = int(np.floor_divide(total_budget, step) + 1)

for e in range(0, N_EXPERIMENTS):
    opt = CMABOptimizer(max_budget=total_budget, campaigns_number=N_CLASSES, step=step)
    env = NSCMABEnvironment(budgets_list=budgets_j, sigma=sigma, horizon=TIME_SPAN)
    gpts_learners = [ GPTS_Learner(n_arms=n_arms[v], arms=budgets_j[v]) for v in range(0, N_CLASSES) ]
    gpsw_learners = [GPSWTS_Learner(n_arms=n_arms[v], arms=budgets_j[v], window_size=WINDOW_SIZE) for v in range(0, N_CLASSES) ]

    for t in range(0, TIME_SPAN):
        # Logger of completion
        if t % int(TIME_SPAN/7) == 0:
            timestampStr = datetime.now().strftime("%H:%M:%S")
            print(timestampStr + " - Step %s of %s (%s exp)" % ((t / int(TIME_SPAN/7)), TIME_SPAN / int(TIME_SPAN/7), e + 1))

        # TS: Create matrix for the optimization process by sampling the GPTS
        ts_base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
        for j in range(0, N_CLASSES):
            ts_sampled_values = gpts_learners[j].sample_values()
            ts_bubblesNum = int(min_budgets[j] / step)
            ts_indices_list = [i for i in range(ts_bubblesNum + colNum * j, ts_bubblesNum + colNum * j + len(ts_sampled_values))]
            np.put(ts_base_matrix, ts_indices_list, ts_sampled_values)

        # SWTS: Create matrix for the optimization process by sampling the GPTS
        sw_base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
        for j in range(0, N_CLASSES):
            sw_sampled_values = gpsw_learners[j].sample_values()
            sw_bubblesNum = int(min_budgets[j] / step)
            sw_indices_list = [i for i in range(sw_bubblesNum + colNum * j, sw_bubblesNum + colNum * j + len(sw_sampled_values))]
            np.put(sw_base_matrix, sw_indices_list, sw_sampled_values)

        # Choose budget thanks to the samples in the matrix
        ts_chosen_budget = opt.optimize(ts_base_matrix)
        sw_chosen_budget = opt.optimize(sw_base_matrix)

        # Update model of the GPTS
        ts_arms_chosen = []
        for j in range(0, N_CLASSES):
            ts_chosen_arm = gpts_learners[j].convert_value_to_arm(ts_chosen_budget[j])
            ts_chosen_arm = int(ts_chosen_arm[0])
            ts_reward = env.round(ts_chosen_arm, j)
            gpts_learners[j].update(ts_chosen_arm, ts_reward)

        # Update model of the GPSWTS
        sw_arms_chosen = []
        for j in range(0, N_CLASSES):
            sw_chosen_arm = gpsw_learners[j].convert_value_to_arm(sw_chosen_budget[j])
            sw_chosen_arm = int(sw_chosen_arm[0])
            sw_reward = env.round(sw_chosen_arm, j)
            gpsw_learners[j].update(sw_chosen_arm, sw_reward)

        env.ahead()

    # Append rewards for statistical purposes
    for j in range(0, N_CLASSES):
        per_experiment_rewards_gpts[j].append(gpts_learners[j].collected_rewards)
        per_experiment_rewards_gpswts[j].append(gpsw_learners[j].collected_rewards)

opt_per_phase = []
for p in range(0, env.N_PHASES):
    # Compute the REAL optimum allocation by solving the optimization problem with the real values
    base_matrix = np.ones((N_CLASSES, colNum)) * np.NINF
    for j in range(0, N_CLASSES):
        real_values = env.means[j*env.N_PHASES + p]
        bubblesNum = int(min_budgets[j] / step)
        indices_list = [i for i in range(bubblesNum + colNum * j, bubblesNum + colNum * j + len(real_values))]
        np.put(base_matrix, indices_list, real_values)
    chosen_budget = opt.optimize(base_matrix)  # Optimal allocation in phase p
    optimal_val = 0
    for j in range(0, N_CLASSES):
        optimal_val += fun(chosen_budget[j], j, p)
    opt_per_phase.append(optimal_val)

# Print aggregated results
aggr_rewards_gpts = np.sum(per_experiment_rewards_gpts, axis=0)
aggr_rewards_gpts = np.mean(aggr_rewards_gpts, axis=0)
aggr_rewards_gpswts = np.sum(per_experiment_rewards_gpswts, axis=0)
aggr_rewards_gpswts = np.mean(aggr_rewards_gpswts, axis=0)

plt.figure(0)
plt.ylabel("Regret [clicks]")
plt.xlabel("Time [days]")
optimal_vector = []
for t in range(0, TIME_SPAN):
    phase_size = TIME_SPAN / env.N_PHASES
    current_phase = int(t / phase_size)
    optimal_vector.append(opt_per_phase[current_phase])
plt.plot(np.cumsum(optimal_vector - aggr_rewards_gpts), 'r')
plt.plot(np.cumsum(optimal_vector - aggr_rewards_gpswts), 'b')
plt.legend(["GPTS", "GP-SWTS"])
plt.show()

plt.figure(1)
plt.ylabel("Reward [clicks]")
plt.xlabel("Time [days]")
plt.plot(aggr_rewards_gpts, 'r')
plt.plot(aggr_rewards_gpswts, 'b')
plt.plot(optimal_vector, 'k--')
plt.legend(["GPTS", "GP-SWTS", "Optimal"])
plt.show()

# Storing results
timestamp = str(datetime.timestamp(datetime.now()))
regretTS = np.cumsum(optimal_vector - aggr_rewards_gpts)
with open(timestamp + "-gpts_regret.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(regretTS)
writeFile.close()

regretSW = np.cumsum(optimal_vector - aggr_rewards_gpswts)
with open(timestamp + "-gp_swts_regret.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(regretSW)
writeFile.close()