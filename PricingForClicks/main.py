import matplotlib.pyplot as plt
from PricingForClicks.Environment import *
from PricingForClicks.TS_Learner import *
from PricingForClicks.UCB1_Learner import *
import utils, math
from scipy import optimize
from scipy.stats import beta, norm


# demand curve
def demand(x):
    return utils.getDemandCurve(-1, x)  # aggregated fn


B = 5000  # avg number of clicks per day
var = 500  # variance in number of clicks per day
T = 300  # number of days

clicks = np.round(np.random.normal(B, var, T))  # num of users who clicked the ads on each day
best_price = optimize.fmin(lambda x: -demand(x) * x, T / 2)[0]
best_demand = demand(best_price) * best_price
print("Best price ", best_price)
optimum = best_demand * clicks

n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
print("Optimal number of arms: %d" % n_arms)

n_experiments = 20

ts_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
ucb1_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
all_ts_regret = []
all_ucb1_regret = []
for e in range(0, n_experiments):
    ts_learner = TS_Learner(arms=ts_env.probabilities)
    ucb1_learner = UCB1_Learner(arms=ucb1_env.probabilities[:, 0])
    ts_regret = []
    ucb1_regret = []
    for t in range(0, T):
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        successes = ts_env.round(pulled_arm, clicks[t])
        failures = clicks[t] - successes
        ts_learner.update(pulled_arm, successes, failures)
        best_hope = optimum[t]
        actual_value = successes*ts_env.probabilities[pulled_arm][0]
        regret = best_hope - actual_value
        ts_regret.append(regret)

        # UCB1 Learner
        pulled_arm = ucb1_learner.pull_arm()
        successes = ucb1_env.round(pulled_arm, clicks[t])
        failures = clicks[t] - successes
        ucb1_learner.update(pulled_arm, successes, failures)
        best_hope = optimum[t]
        actual_value = successes * ucb1_env.probabilities[pulled_arm, 0]
        regret = best_hope - actual_value
        ucb1_regret.append(regret)

    all_ts_regret.append(ts_regret)
    all_ucb1_regret.append(ucb1_regret)

plt.figure(0)
plt.plot(np.cumsum(np.mean(all_ts_regret, axis=0)))
plt.plot(np.cumsum(np.mean(all_ucb1_regret, axis=0)))


plt.legend(["TS", "UCB1"])
plt.xlabel("T")
plt.ylabel("Regret[€]")
plt.show()

x = np.arange(0, 1, 0.01)
plt.figure(1)
for i in range(n_arms):
    [a, b] = ts_learner.beta_parameters[i]
    y = beta.pdf(x, a, b)
    plt.plot(x, y)

plt.legend(range(n_arms))
plt.xlabel("")
plt.ylabel("beta distribution of arms")
plt.show()

plt.figure(2)

for i in range(n_arms):
    (avg, n, price) = ucb1_learner.results_per_arm[i]
    sigma = ucb1_learner.calc_upper_bound((avg, n, price)) / price - avg
    y = norm.pdf(x, avg, sigma)
    plt.plot(x, y)

plt.ylabel("UCB1 average and upper bound")
plt.legend(range(n_arms))
plt.show()