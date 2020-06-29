import matplotlib.pyplot as plt
from PricingForClicks.PricingForClicks3.Environment import *
from PricingForClicks.PricingForClicks3.TS_Learner import *
from PricingForClicks.PricingForClicks3.UCB1_Learner import *
import utils

# avg number of clicks per day, taken from the best budget allocation of the previous point
B = np.array([1800, 12000, 350])
var = B / 4  # variance in number of clicks per day
T = 50  # number of days
multi_arms = True
N_CLASSES = 3
CLASSES = range(N_CLASSES)

demand = [lambda t, c=c: utils.getDemandCurve(c, t) for c in CLASSES]

clicks = [np.round(np.random.normal(B[c], var[c], T)) for c in CLASSES]  # num of users who clicked the ads on each day

best_n_arms = 4  # math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms

print("Optimal number of arms: %d" % best_n_arms)

n_experiments = 1000
MIN_N_ARMS = 3
MAX_N_ARMS = 10
MIN_PRICE = 100
MAX_PRICE = 400
n_arms_arr = list(range(MIN_N_ARMS, MAX_N_ARMS)) if multi_arms else [best_n_arms]

max_regret_per_arm = []

x = np.arange(MIN_PRICE, MAX_PRICE, 1)
real_best_prices = np.array([np.argmax(demand[c](x) * x) for c in CLASSES]) + MIN_PRICE
print("Theoretical best prices: %s" % str(real_best_prices))
real_optimum = [demand[c](real_best_prices[c]) * real_best_prices[c] * clicks[c] for c in CLASSES]

ts_regret_per_arm = []
ts_reward_per_arm = []
for n_arms in n_arms_arr:
    print(n_arms, "arms")
    ts_env = [Environment(n_arms=n_arms, demandCurve=demand[c], minPrice=MIN_PRICE, maxPrice=MAX_PRICE) for c in CLASSES]
    prices = [t.probabilities[:, 0] for t in ts_env]
    best_arms = [np.argmax(demand[c](prices[c]) * prices[c]) for c in CLASSES]
    best_prices = [prices[c][best_arms[c]] for c in CLASSES]
    print("Theoretical best arms for %d arms: %s" % (n_arms, str(best_arms)))
    print("Theoretical best prices for %d arms: %s" % (n_arms, str(best_prices)))
    optimum = [demand[j](best_prices[j]) * best_prices[j] * clicks[j] for j in CLASSES]

    per_experiment_regret = []
    per_experiment_reward = []
    for e in range(0, n_experiments):
        ts_learners = [TS_Learner(arms=t.probabilities) for t in ts_env]
        daily_regret = []
        daily_reward = []
        for t in range(0, T):
            ts_regret_per_class = []
            ts_reward_per_class = []
            for userType in CLASSES:
                # Thompson Sampling Learner
                pulled_arm = ts_learners[userType].pull_arm()
                pulled_price = ts_env[userType].probabilities[pulled_arm][0]
                successes = ts_env[userType].round(pulled_arm, clicks[userType][t])
                failures = clicks[userType][t] - successes
                ts_learners[userType].update(pulled_arm, successes, failures)
                reward = successes * pulled_price
                regret = optimum[userType][t] - reward
                # append normalized reward
                ts_reward_per_class.append(reward) # / clicks[userType][t])
                ts_regret_per_class.append(regret)

            daily_regret.append(ts_regret_per_class)
            daily_reward.append(ts_reward_per_class)

        per_experiment_regret.append(daily_regret)
        per_experiment_reward.append(daily_reward)

    tot_regret = np.cumsum(np.mean(np.sum(per_experiment_regret, axis=2), axis=0))
    tot_reward = np.mean(np.sum(per_experiment_reward, axis=2), axis=0)
    max_regret_per_arm.append(tot_regret[len(tot_regret) - 1])
    ts_regret_per_arm.append(tot_regret)
    ts_reward_per_arm.append(tot_reward)

tmp = np.argsort(max_regret_per_arm) + MIN_N_ARMS
print("regrets per arm: %s" % str(np.sort(max_regret_per_arm)))
print("Sorted arms: %s" % str(tmp))
print("With %d arms we have the smallest regret: %d" % (MIN_N_ARMS + np.argmin(max_regret_per_arm), int(np.min(max_regret_per_arm))))

plt.figure(0)
for n in range(len(n_arms_arr)):
    plt.plot(ts_regret_per_arm[n])
plt.legend(n_arms_arr)
plt.xlabel("T [days]")
plt.ylabel("Regret[€], TS")
plt.show()

plt.figure(1)
plt.plot(np.sum(real_optimum, axis=0), '--')
for n in range(len(n_arms_arr)):
    plt.plot(ts_reward_per_arm[n])
legend = ["clairvoyant"] + n_arms_arr
plt.legend(legend)
plt.xlabel("T")
plt.ylabel("Reward[€], TS")
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.sum(real_optimum, axis=0)), '--')
cumulative_rewards = []
for n in range(len(n_arms_arr)):
    c = np.cumsum(ts_reward_per_arm[n])
    cumulative_rewards.append(c[len(c)-1])
    plt.plot(c)
plt.legend(legend)
plt.xlabel("T [days]")
plt.ylabel("Cumulative Reward[€]")
plt.show()

print("Cumulative rewards: %s" % str(cumulative_rewards))
print("Arms sorted by reward: %s" % str(np.flip(np.argsort(cumulative_rewards)) + MIN_N_ARMS))
