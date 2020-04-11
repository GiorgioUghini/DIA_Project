import matplotlib.pyplot as plt
import utils as u
from context_generator.TSContextGenerator import *
from datetime import datetime
import csv

regret = []
number_of_arms = 6
number_of_experiments = 5
number_of_clicks = 280000   # 20 days
interval_for_splitting = 14000   # 2000 clicks per day
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
            timestampStr = datetime.now().strftime("%H:%M:%S")
            print(timestampStr + " - Step %s of %s (%s exp)" % ((i/interval_for_splitting)+1, number_of_clicks/interval_for_splitting, e+1))
            ts_context_generator.split()

    regret.append(ts_context_generator.regrets)

plt.figure(0)
plt.plot(np.cumsum(np.mean(regret, axis=0)))
plt.legend(["6 arms - TS"])
plt.xlabel("number of clicks")
plt.ylabel("regret")
plt.show()

# Storing results
time = datetime.now().strftime("%d-%m-%Y-at-%H-%M-%S")
proc_regret = np.mean(regret, axis=0)
with open(time + "-regret.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(proc_regret)
writeFile.close()
