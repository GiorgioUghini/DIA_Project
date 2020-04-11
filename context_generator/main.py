import matplotlib.pyplot as plt
import utils as u
from context_generator.TSContextGenerator import *

regret = []
number_of_arms = 6
number_of_experiments = 10
number_of_clicks = 14000
interval_for_splitting = 700
probabilities_of_users = [u.getProbabilities(0), u.getProbabilities(1), u.getProbabilities(2)]
arms = np.array([100 + ((400-100)/number_of_arms) * i for i in range(number_of_arms)])
arms_demand_for_each_user = np.zeros([number_of_arms, len(probabilities_of_users)])

for i in range(number_of_arms):
    for j in range(len(probabilities_of_users)):
        arms_demand_for_each_user[i][j] = u.getDemandCurve(j, arms[i])

for e in range(number_of_experiments):
    ts_context_generator = TSContextGenerator(probabilities_of_users, arms_demand_for_each_user, arms)
    for i in range(number_of_clicks):
        ts_context_generator.calc_clair_rew_regret()
        if i % interval_for_splitting == 0:
            print("Step %s of %s (%s exp)" % (i/interval_for_splitting, number_of_clicks/interval_for_splitting, e))
            ts_context_generator.split()

    regret.append(ts_context_generator.regrets)

plt.figure(0)
plt.plot(np.cumsum(np.mean(regret, axis=0)))
plt.legend(["6 arms - TS"])
plt.xlabel("number of clicks")
plt.ylabel("regret")
plt.show()
