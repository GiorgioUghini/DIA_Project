import matplotlib.pyplot as plt
from PricingForClicks.PricingForClicks2.Environment import *
from PricingForClicks.PricingForClicks2.TS_Learner import *
from PricingForClicks.PricingForClicks2.UCB1_Learner import *
import utils, math
from scipy.stats import beta, norm


# demand curve
def demand(x):
    return utils.getDemandCurve(-1, x)  # aggregated fn


B = 200  # avg number of clicks per day
var = 20  # variance in number of clicks per day
days = 14  # number of days
multi_arms = False

clicks = (np.round(np.random.normal(B, var, days)))  # num of users who clicked the ads on each day

T = days * np.sum(clicks)

best_n_arms = math.ceil((T * np.log10(T)) ** 0.25)  # the optimal number of arms
best_beta_params = []  # beta parameters of the optimal number of arms
best_ucb1_params = []  # parameters for the optimal number of arms

print("Optimal number of arms: %d" % best_n_arms)

n_experiments = 100
n_arms_arr = range(best_n_arms-3, best_n_arms + 4) if multi_arms else [best_n_arms]

plt.figure(0)
for n_arms in n_arms_arr:
    print(n_arms, "arms")
    ts_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
    ucb_env = Environment(n_arms=n_arms, demandCurve=demand, minPrice=0, maxPrice=400)
    prices = ts_env.probabilities[:, 0]
    best_arm = np.argmax(demand(prices) * prices)
    best_price = prices[best_arm]
    print("Theoretical best arm for %d arms: %d" % (n_arms, best_arm))
    print("Theoretical best price for %d arms: %d" % (n_arms, best_price))
    optimum = demand(best_price) * best_price

    ts_reward_per_experiment = []
    ucb_reward_per_experiment = []
    for e in range(0, n_experiments):
        print("Experiment", e, "T:", T)
        ts_learner = TS_Learner(arms=ts_env.probabilities)
        ucb_learner = UCB1_Learner(arms=ucb_env.probabilities)
        ts_regret = []
        ucb_regret = []
        ts_rewards = []
        ucb_rewards = []
        for t in range(0, np.int(T)):
            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            result = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, result)
            ts_rewards.append(result*ts_env.probabilities[pulled_arm][0])

            # UCB1 learner
            pulled_arm = ucb_learner.pull_arm()
            result = ucb_env.round(pulled_arm)
            ucb_learner.update(pulled_arm, result)
            ucb_rewards.append(result*ucb_env.probabilities[pulled_arm][0])

        ts_reward_per_experiment.append(ts_rewards)
        ucb_reward_per_experiment.append(ucb_rewards)

    if n_arms == best_n_arms:
        best_beta_params = ts_learner.beta_parameters

    plt.figure(0)
    plt.plot(np.cumsum(np.mean(optimum - ts_reward_per_experiment, axis=0)))
    plt.figure(1)
    plt.plot(np.cumsum(np.mean(optimum - ucb_reward_per_experiment, axis=0)))

plt.figure(0)
plt.legend(n_arms_arr)
plt.xlabel("Clicks")
plt.ylabel("Regret[€], TS")

plt.figure(1)
plt.legend(n_arms_arr)
plt.xlabel("Clicks")
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
