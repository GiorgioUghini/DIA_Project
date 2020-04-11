from context_generator.Context import *
from context_generator.SplitFailedEx import *


class TSContextGenerator:

    def __init__(self, probabilities_of_users, arms_demand_for_each_user, arms):
        self.probabilities_of_users = probabilities_of_users
        self.arms_demand_for_each_user = arms_demand_for_each_user
        self.arms = arms
        self.realizations_per_arm_per_demand = [[
            [] for _ in range(len(probabilities_of_users))
        ] for _ in range(len(arms))]
        self.beta_params = np.ones([len(arms), len(probabilities_of_users), 2], dtype=np.int64)
        self.regrets = []
        self.contexts = [Context([i for i in range(len(probabilities_of_users))])]

    def n(self, arm, demand):
        return len(self.realizations_per_arm_per_demand[arm][demand])

    def emp_mean(self, arm, demand):
        return np.mean(self.realizations_per_arm_per_demand[arm][demand])

    def calc_clair_rew_regret(self):
        clairvoyant = 0
        reward = 0
        for i in range(len(self.contexts)):
            context = self.contexts[i]
            clairvoyant += context.get_clairvoyant(self.probabilities_of_users, self.arms_demand_for_each_user, self.arms)
            best_arm = context.get_best_arm(self.probabilities_of_users, self, self.arms)

            for demand in context.demands_index_list:
                sort = np.random.binomial(n=1, size=1, p=self.arms_demand_for_each_user[best_arm][demand])[0]
                self.realizations_per_arm_per_demand[best_arm][demand].append(sort)
                self.beta_params[best_arm][demand] += np.array([sort, 1 - sort])

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
