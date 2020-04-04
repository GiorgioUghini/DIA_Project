import matplotlib.pyplot as plt
from PricingForClicks.Environment import *
from PricingForClicks.TS_Learner import *
from PricingForClicks.UCB1_Learner import *
import utils, math
from scipy import optimize


# demand curve
def demand(x):
    return utils.getDemandCurve(-1, x)  # aggregated fn


B = 6000  # budget
var = 9  # variance (where do I take this from?)
T = 300  # number of days

clicks = np.round(np.random.normal(B, var, T))  # num of users who clicked the ads on each day
best_price = optimize.fmin(lambda x: -demand(x), T / 2)
optimum = demand(best_price) * clicks

n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
print("Optimal number of arms: %d" % n_arms)

n_arms_arr = []
colors = ['r', 'g', 'b', 'c', 'y']

for n in range(n_arms-2, n_arms + 3):
    n_arms_arr.append((n, colors[n % 5]))
n_arms_arr.append((12, 'k--'))  # 25€ steps

np_arms_arr = np.array(n_arms_arr)

n_experiments = 50
ts_rewards_per_experiment = []
legend = []

print("N ARMS: %d" % n_arms)
ts_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=100, maxPrice=300)
ucb1_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=100, maxPrice=300)
all_ts_regret = []
all_ucb1_regret = []
for e in range(0, n_experiments):
    ts_learner = TS_Learner(n_arms=n_arms)
    ucb1_learner = UCB1_Learner(n_arms=n_arms)
    ts_regret = []
    ucb1_regret = []
    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        successes = ts_env.round(pulled_arm, clicks[t])
        failures = clicks[t] - successes
        ts_learner.update(pulled_arm, successes, failures)
        ts_regret.append(clicks[t]*demand(best_price) - successes)

        # UCB1 Learner
        pulled_arm = ucb1_learner.pull_arm()
        successes = ucb1_env.round(pulled_arm, clicks[t])
        failures = clicks[t] - successes
        ucb1_learner.update(pulled_arm, successes, failures)
        ucb1_regret.append(clicks[t] * demand(best_price) - successes)

    all_ts_regret.append(ts_regret)
    all_ucb1_regret.append(ucb1_regret)

plt.figure(0)
plt.plot(np.cumsum(np.mean(all_ts_regret, axis=0)), "r")
plt.plot(np.cumsum(np.mean(all_ucb1_regret, axis=0)), "b")
legend = ["TS", "UCB1"]

plt.legend(legend)
plt.show()
