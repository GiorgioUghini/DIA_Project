import matplotlib.pyplot as plt
from PricingForClicks.PricingForClicks3.Environment import *
from PricingForClicks.PricingForClicks3.TS_Learner import *
from PricingForClicks.PricingForClicks3.UCB1_Learner import *
import utils

# avg number of clicks per day, taken from the best budget allocation of the previous point
B = np.array([1800, 12000, 350])
var = B / 4  # variance in number of clicks per day
T = 45  # number of days
multi_arms = True
N_CLASSES = 3
CLASSES = range(N_CLASSES)

demand = [lambda t, c=c: utils.getDemandCurve(c, t) for c in CLASSES]

clicks = [np.round(np.random.normal(B[c], var[c], T)) for c in CLASSES]  # num of users who clicked the ads on each day

best_n_arms = 4  # math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
best_beta_params = []  # beta parameters of the optimal number of arms
best_ucb1_params = []  # parameters for the optimal number of arms

print("Optimal number of arms: %d" % best_n_arms)

n_experiments = 10000
n_arms_arr = range(4, 10) if multi_arms else [best_n_arms]

max_regret_per_arm = []

x = np.arange(0, 400, 1)
real_best_prices = [np.argmax(demand[c](x) * x) for c in CLASSES]
real_optimum = [demand[c](real_best_prices[c]) * real_best_prices[c] * clicks[c] for c in CLASSES]

ts_regret_per_arm = []
ucb1_regret_per_arm = []
for n_arms in n_arms_arr:
    print(n_arms, "arms")
    ts_env = [Environment(n_arms=n_arms, demandCurve=demand[c], minPrice=0, maxPrice=400) for c in CLASSES]
    ucb1_env = [Environment(n_arms=n_arms, demandCurve=demand[c], minPrice=0, maxPrice=400) for c in CLASSES]
    prices = [t.probabilities[:, 0] for t in ts_env]
    best_arms = [np.argmax(demand[c](prices[c]) * prices[c]) for c in CLASSES]
    best_prices = [prices[c][best_arms[c]] for c in CLASSES]
    print("Theoretical best arms for %d arms: %s" % (n_arms, str(best_arms)))
    print("Theoretical best prices for %d arms: %s" % (n_arms, str(best_prices)))
    optimum = [demand[j](best_prices[j]) * best_prices[j] * clicks[j] for j in CLASSES]

    all_ts_regret = []
    all_ucb1_regret = []
    for e in range(0, n_experiments):
        ts_learners = [TS_Learner(arms=t.probabilities) for t in ts_env]
        ucb1_learners = [UCB1_Learner(arms=u.probabilities[:, 0]) for u in ucb1_env]
        ts_regret = []
        ucb1_regret = []
        for t in range(0, T):
            ts_regret_per_class = []
            ucb1_regret_per_class = []
            for userType in CLASSES:
                # Thompson Sampling Learner
                pulled_arm = ts_learners[userType].pull_arm()
                successes = ts_env[userType].round(pulled_arm, clicks[userType][t])
                failures = clicks[userType][t] - successes
                ts_learners[userType].update(pulled_arm, successes, failures)
                actual_value = successes*ts_env[userType].probabilities[pulled_arm][0]
                regret = real_optimum[userType][t] - actual_value
                ts_regret_per_class.append(regret)

                # UCB1 Learner
                pulled_arm = ucb1_learners[userType].pull_arm()
                successes = ucb1_env[userType].round(pulled_arm, clicks[userType][t])
                failures = clicks[userType][t] - successes
                ucb1_learners[userType].update(pulled_arm, successes, failures)
                actual_value = successes * ucb1_env[userType].probabilities[pulled_arm, 0]
                regret = real_optimum[userType][t] - actual_value
                ucb1_regret_per_class.append(regret)

            ts_regret.append(ts_regret_per_class)
            ucb1_regret.append(ucb1_regret_per_class)

        all_ts_regret.append(ts_regret)
        all_ucb1_regret.append(ucb1_regret)

    if n_arms == best_n_arms:
        best_beta_params = [ts_learners[userType].beta_parameters for userType in CLASSES]
        best_ucb1_params = [ucb1_learners[userType].results_per_arm for userType in CLASSES]

    ts_regret = np.cumsum(np.mean(np.sum(all_ts_regret, axis=2), axis=0))
    max_regret_per_arm.append(ts_regret[len(ts_regret)-1])
    ts_regret_per_arm.append(ts_regret)
    ucb1_regret_per_arm.append(np.cumsum(np.mean(np.sum(all_ucb1_regret, axis=2), axis=0)))


tmp = np.argsort(max_regret_per_arm) + 4
print("regrets per arm: %s" % str(np.sort(max_regret_per_arm)))
print("Sorted arms: %s" % str(tmp))
print("With %d arms we have the smallest regret: %d" % (4 + np.argmin(max_regret_per_arm), int(np.min(max_regret_per_arm))))

plt.figure(0)
for n in range(len(n_arms_arr)):
    plt.plot(ts_regret_per_arm[n])
plt.legend(n_arms_arr)
plt.xlabel("T")
plt.ylabel("Regret[€], TS")
plt.show()
plt.figure(1)
for n in range(len(n_arms_arr)):
    plt.plot(ucb1_regret_per_arm[n])
plt.legend(n_arms_arr)
plt.xlabel("T")
plt.ylabel("Regret[€], UCB1")
plt.show()
