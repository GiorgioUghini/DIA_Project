import matplotlib.pyplot as plt
import utils as u
from context_generator.TSContextGenerator import *
from datetime import datetime
import csv

regret = []
number_of_arms = 6
number_of_experiments = 10
clicks_per_day = 2000
interval_for_splitting = 7
number_of_days = 140
probabilities_of_users = [u.getProbabilities(0), u.getProbabilities(1), u.getProbabilities(2)]
arms = np.array([100 + ((400-100)/number_of_arms) * i for i in range(number_of_arms)])
arms_demand_for_each_user = np.zeros([number_of_arms, len(probabilities_of_users)])

for i in range(number_of_arms):
    for j in range(len(probabilities_of_users)):
        arms_demand_for_each_user[i][j] = u.getDemandCurve(j, arms[i])

for e in range(number_of_experiments):
    print(datetime.now().strftime("%H:%M:%S") + " - %s experiment beginning" % (e+1))
    ts_context_generator = TSContextGenerator(probabilities_of_users, arms_demand_for_each_user, arms)
    for i in range(number_of_days):
        ts_context_generator.calc_clair_rew_regret(clicks_per_day)
        if i % interval_for_splitting == 0  and i != 0:
            ts_context_generator.split()

    regret.append(ts_context_generator.regrets)

plt.figure(0)
plt.plot(np.cumsum(np.mean(regret, axis=0)),)
plt.legend(["6 arms - TS"])
plt.xlabel("number of days")
plt.ylabel("regret")
plt.xticks(np.linspace(0, clicks_per_day*number_of_days, 8), np.linspace(0, number_of_days, 8, dtype=np.int32))
plt.show()

# Storing results
"""
time = datetime.now().strftime("%d-%m-%Y-at-%H-%M-%S")
proc_regret = np.mean(regret, axis=0)
with open(time + "-regret.csv", "w") as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(proc_regret)
writeFile.close()
"""