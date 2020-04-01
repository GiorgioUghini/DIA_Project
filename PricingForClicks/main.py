import matplotlib.pyplot as plt
from PricingForClicks.Environment import *
from PricingForClicks.TS_Learner import *
import utils, math
from scipy import optimize


f = lambda x: utils.getDemandCurve(0, x)
T = 300
best_x = optimize.fmin(lambda x: -f(x), T/2)
optimum = f(best_x)

n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
print("Optimal number of arms: %d" % n_arms)

n_arms_arr = []
colors = ['r', 'g', 'b', 'c', 'y']

for n in range(n_arms-2, n_arms + 3):
    n_arms_arr.append((n, colors[n % 5]))

np_arms_arr = np.array(n_arms_arr)

n_experiments = 1000
ts_rewards_per_experiment = []

for (n_arms, c) in n_arms_arr:
    print("N ARMS: %d" % n_arms)
    env = Environment(n_arms=n_arms, demandCurve=f, minPrice=0, maxPrice=400)
    for e in range(0, n_experiments):
        ts_learner = TS_Learner(n_arms=n_arms)
        for t in range(0, T):
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)

    plt.figure(0)
    plt.plot(np.cumsum(np.mean(optimum - ts_rewards_per_experiment,axis=0)), c)

plt.legend(np_arms_arr[:, 1])
plt.show()
