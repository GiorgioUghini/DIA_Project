import matplotlib.pyplot as plt
from PricingForClicks.Environment import *
from PricingForClicks.TS_Learner import *
import utils, math
from scipy import optimize


# demand curve
def demand(x):
    return utils.getDemandCurve(-1, x)  # aggregated fn


B = 300  # budget
var = 4  # variance (where do I take this from?)
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

n_experiments = 10
ts_rewards_per_experiment = []
legend = []

for (n_arms, c) in n_arms_arr:
    print("N ARMS: %d" % n_arms)
    env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=100, maxPrice=400)
    all_regrets = []
    for e in range(0, n_experiments):
        ts_learner = TS_Learner(n_arms=n_arms)
        regret = []
        for t in range(0, T):
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            successes = env.round(pulled_arm, clicks[t])
            failures = clicks[t] - successes
            ts_learner.update(pulled_arm, successes, failures)
            #regret.append(optimum[t] * best_price - successes * env.probabilities[pulled_arm][0])
            regret.append(clicks[t]*demand(best_price) - successes)


        all_regrets.append(regret)
        #ts_rewards_per_experiment.append(ts_learner.collected_rewards)

    plt.figure(0)
    tmp = np.cumsum(np.mean(all_regrets, axis=0))
    plt.plot(tmp, c)
    legend.append(str(n_arms) + " arms")

plt.legend(legend)
plt.show()
