import matplotlib.pyplot as plt
import utils as u
from context_generator.TSContextGenerator import *
from datetime import datetime

regret = []
reward = []
clair = []
number_of_arms = 6
number_of_experiments = 1
number_of_clicks_per_day = 20
number_of_days = 140
number_of_days_for_splitting = 7
probabilities_of_users = [u.getProbabilities(0), u.getProbabilities(1), u.getProbabilities(2)]
arms = np.array([100 + ((400-100)/number_of_arms) * i for i in range(number_of_arms)])
arms_samples_for_each_user = np.zeros([number_of_arms, len(probabilities_of_users)])

for i in range(number_of_arms):
    for j in range(len(probabilities_of_users)):
        arms_samples_for_each_user[i][j] = u.getDemandCurve(j, arms[i])

for e in range(number_of_experiments):
    print(datetime.now().strftime("%H:%M:%S") + " - %s experiment beginning" % (e+1))
    ts_context_generator = TSContextGenerator(probabilities_of_users, arms_samples_for_each_user, arms)
    for i in range(number_of_days):
        ts_context_generator.update_regret_after_day_passed(number_of_clicks_per_day)
        if i % number_of_days_for_splitting == 0 and i != 0:
            ts_context_generator.split()

    regret.append(ts_context_generator.regrets)
    reward.append(ts_context_generator.rewards)
    clair.append(ts_context_generator.clairs)

plt.figure(0)
plt.plot(np.cumsum(np.mean(regret, axis=0)))
plt.legend(["4 arms - TS"])
plt.xlabel("number of days")
plt.ylabel("cumulative regret")
plt.xticks(np.linspace(0, number_of_clicks_per_day*number_of_days, 8), np.linspace(0, number_of_days, 8, dtype=np.int32))
plt.show()
plt.figure(1)
plt.plot(u.smooth(np.mean(reward, axis=0), 800))
plt.plot(u.smooth(np.mean(clair, axis=0), 800))
plt.legend(["reward", "clairvoyant"])
plt.xlabel("number of days")
plt.ylabel("reward")
plt.xticks(np.linspace(0, number_of_clicks_per_day*number_of_days, 8), np.linspace(0, number_of_days, 8, dtype=np.int32))
plt.show()
