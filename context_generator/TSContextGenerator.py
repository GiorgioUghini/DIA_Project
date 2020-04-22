from context_generator.Context import *
from context_generator.SplitFailedEx import *


class TSContextGenerator:

    def __init__(self, probabilities_of_users, arms_demand_for_each_user, arms):
        self.probabilities_of_users = probabilities_of_users
        self.arms_demand_for_each_user = arms_demand_for_each_user
        self.arms = arms
        self.realizations_per_arm_per_demand = [[
            [0, 0] for _ in range(len(probabilities_of_users))
        ] for _ in range(len(arms))]
        self.regrets = []
        self.contexts = [Context([i for i in range(len(probabilities_of_users))])]

    def n(self, arm, demand):
        return self.realizations_per_arm_per_demand[arm][demand][0] + self.realizations_per_arm_per_demand[arm][demand][1]

    def emp_mean(self, arm, demand):
        if self.realizations_per_arm_per_demand[arm][demand][0] == 0:
            return 0
        else:
            return self.realizations_per_arm_per_demand[arm][demand][0] / (self.realizations_per_arm_per_demand[arm][demand][0] + self.realizations_per_arm_per_demand[arm][demand][1])

    def calc_clair_rew_regret(self, clicks_per_day):
        for _ in range(0, clicks_per_day):
            clairvoyant = 0
            reward = 0
            for i in range(len(self.contexts)):
                context = self.contexts[i]
                clairvoyant += context.get_clairvoyant(self.probabilities_of_users, self.arms_demand_for_each_user, self.arms)
                best_arm = context.get_best_arm(self.probabilities_of_users, self, self.arms)

                for demand in context.demands_index_list:
                    sort = np.random.binomial(n=1, size=1, p=self.arms_demand_for_each_user[best_arm][demand])[0]
                    self.realizations_per_arm_per_demand[best_arm][demand][0] += sort
                    self.realizations_per_arm_per_demand[best_arm][demand][1] += 1 - sort

                reward += context.get_arm_reward(self.probabilities_of_users, self, best_arm, self.arms)

            reward = min(reward, clairvoyant)
            self.regrets.append(clairvoyant - reward)

    def split(self):
        for i in range(len(self.contexts)):
            try:
                c1, c2 = self.contexts[i].split(self.probabilities_of_users, self, self.arms)
                self.contexts.pop(i)
                self.contexts.append(c1)
                self.contexts.append(c2)
                break
            except SplitFailedEx:
                continue
