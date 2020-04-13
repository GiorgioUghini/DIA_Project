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


B = 200  # avg number of clicks per day
var = 20  # variance in number of clicks per day
T = 300  # number of days
multi_arms = True

clicks = np.round(np.random.normal(B, var, T))  # num of users who clicked the ads on each day

best_n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
best_beta_params = []  # beta parameters of the optimal number of arms
best_ucb1_params = []  # parameters for the optimal number of arms

print("Optimal number of arms: %d" % best_n_arms)

n_experiments = 1000
n_arms_arr = range(best_n_arms-3, best_n_arms + 4) if multi_arms else [best_n_arms]

for n_arms in n_arms_arr:
    print(n_arms, "arms")
    ts_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
    ucb1_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
    prices = ts_env.probabilities[:, 0]
    best_arm = np.argmax(demand(prices) * prices)
    best_price = prices[best_arm]
    print("Theoretical best arm for %d arms: %d" % (n_arms, best_arm))
    print("Theoretical best price for %d arms: %d" % (n_arms, best_price))
    optimum = demand(best_price) * best_price * clicks

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

    if n_arms == best_n_arms:
        best_beta_params = ts_learner.beta_parameters
        best_ucb1_params = ucb1_learner.results_per_arm

    plt.figure(0)
    plt.plot(np.cumsum(np.mean(all_ts_regret, axis=0)))
    plt.figure(1)
    plt.plot(np.cumsum(np.mean(all_ucb1_regret, axis=0)))

plt.figure(0)
plt.legend(n_arms_arr)
plt.xlabel("T")
plt.ylabel("Regret[€], TS")
plt.figure(1)
plt.legend(n_arms_arr)
plt.xlabel("T")
plt.ylabel("Regret[€], UCB1")
plt.show()

plt.figure(2)
x = np.arange(0, 1, 0.01)
for i in range(best_n_arms):
    [a, b] = best_beta_params[i]
    y = beta.pdf(x, a, b)
    plt.plot(x, y)

plt.legend(range(best_n_arms))
plt.xlabel("")
plt.ylabel("beta distribution of arms")
plt.show()

plt.figure(3)

for i in range(best_n_arms):
    (avg, n, price) = best_ucb1_params[i]
    if not price == 0:
        sigma = ucb1_learner.calc_upper_bound((avg, n, price)) / price - avg
        y = norm.pdf(x, avg, sigma)
        plt.plot(x, y)

plt.ylabel("UCB1 average and upper bound")
plt.legend(range(best_n_arms))
plt.show()
